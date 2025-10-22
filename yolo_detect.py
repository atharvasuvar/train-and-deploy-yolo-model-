import cv2
import os
import time
import numpy as np
import argparse
from ultralytics import YOLO
import pyfirmata

# ========================== ARGUMENT PARSER ==========================
parser = argparse.ArgumentParser(description="Weapon & Threat Detection (Combined YOLO + Arduino)")
parser.add_argument("--model", type=str, default="my_model.pt",
                    help="Path to your combined YOLO model")
parser.add_argument("--source", type=str, default="0", help="0 for webcam or path to video/images")
parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
parser.add_argument("--record", action="store_true", help="Enable video recording")
parser.add_argument("--output", type=str, default="output.avi", help="Output video file")
args = parser.parse_args()

# ========================== CONFIGURATION ==========================
MODEL_PATH = os.path.join(os.getcwd(), args.model)
CONF_THRESHOLD = args.conf
SOURCE = args.source
RECORD = args.record
RECORD_NAME = args.output
RESOLUTION = (640, 480)

# ========================== VERIFY MODEL ==========================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

# ========================== LOAD MODEL ==========================
model = YOLO(MODEL_PATH)
labels = model.names

# ========================== ARDUINO SETUP ==========================
port = "COM9"
try:
    board = pyfirmata.Arduino(port, baudrate=57600)
    servo_pinX = board.get_pin('d:9:s')
    servo_pinY = board.get_pin('d:10:s')
    print("[INFO] Arduino connected successfully!")
except:
    print(f"[ERROR] Cannot connect Arduino on {port}.")
    board = None
    servo_pinX = None
    servo_pinY = None

servoPos = [90, 90]

# ========================== VIDEO SOURCE ==========================
if SOURCE == "0":
    SOURCE = 0
    source_type = 'webcam'
elif os.path.isfile(SOURCE):
    source_type = 'video'
elif os.path.isdir(SOURCE):
    source_type = 'folder'
else:
    raise ValueError("Invalid source")

if source_type in ['webcam', 'video']:
    cap = cv2.VideoCapture(SOURCE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
    cap.set(cv2.CAP_PROP_FPS, 60)
    if RECORD:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(RECORD_NAME, fourcc, 60, RESOLUTION)
else:
    cap = None

# ========================== COLORS ==========================
colors = {'gun': (0,0,255), 'knife': (0,165,255), 'threat': (0,0,255), 'safe': (0,255,0)}
ws, hs = RESOLUTION
fps_buffer, fps_len = [], 30

# ========================== PROCESS FRAME ==========================
def process_frame(frame):
    results = model(frame, verbose=False, imgsz=640)[0]
    counts = {'gun':0,'knife':0,'threat':0,'safe':0}

    for box in results.boxes:
        cls_id = int(box.cls.item())
        label = labels[cls_id].lower()
        conf = float(box.conf.item())
        if conf < CONF_THRESHOLD:
            continue
        x1,y1,x2,y2 = map(int, box.xyxy.cpu().numpy().squeeze())

        # Determine color based on label
        if label in ['gun','knife']:
            color = colors[label]
        elif label in ['mask','cover','threat']:
            color = colors['threat']
            label = 'THREAT'
        else:
            color = colors['safe']
            label = 'SAFE'

        counts[label.lower()] = counts.get(label.lower(),0)+1
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,3)
        cv2.putText(frame,f"{label} {int(conf*100)}%",(x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)

        # Servo tracking first face
        if label in ['THREAT','SAFE']:
            fx, fy = (x1+x2)//2, (y1+y2)//2
            servoX = np.interp(fx,[0,ws],[180,0])
            servoY = np.interp(fy,[0,hs],[180,0])
            servoPos[0], servoPos[1] = servoX, servoY
            if servo_pinX and servo_pinY:
                servo_pinX.write(servoX)
                servo_pinY.write(servoY)

    return frame, counts

# ========================== MAIN LOOP ==========================
if source_type in ['webcam','video']:
    while True:
        ret, frame = cap.read()
        if not ret: break
        start = time.perf_counter()
        frame, counts = process_frame(frame)
        end = time.perf_counter()
        fps = 1/(end-start)
        fps_buffer.append(fps)
        if len(fps_buffer)>fps_len: fps_buffer.pop(0)
        avg_fps = np.mean(fps_buffer)

        # Overlay info
        cv2.putText(frame,f"FPS: {avg_fps:.2f}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
        cv2.putText(frame,f"Guns: {counts.get('gun',0)} Knives: {counts.get('knife',0)}",(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
        cv2.putText(frame,f"THREAT: {counts.get('threat',0)} SAFE: {counts.get('safe',0)}",(10,90),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
        cv2.imshow("YOLO - Combined Detection",frame)
        if RECORD: out.write(frame)
        if cv2.waitKey(1)&0xFF==ord('q'): break

    cap.release()
    if RECORD: out.release()
    cv2.destroyAllWindows()
