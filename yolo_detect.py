import cv2
import os
import time
import numpy as np
import argparse
from ultralytics import YOLO
import pyfirmata

# ==========================
# ARGUMENT PARSER
# ==========================
parser = argparse.ArgumentParser(description="Weapon & Mask Threat Detection (YOLO + Arduino)")
parser.add_argument("--weapon_model", type=str, default="runs/detect/train/weights/best.pt",
                    help="Path to your trained weapon YOLO model")
parser.add_argument("--mask_model", type=str, default="runs/detect/mask/weights/best.pt",
                    help="Path to your trained mask YOLO model")
parser.add_argument("--source", type=str, default="0",
                    help="Source: 0 for webcam, video file path, or image folder path")
parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
parser.add_argument("--record", action="store_true", help="Enable video recording")
parser.add_argument("--output", type=str, default="output.avi", help="Output video file name")
args = parser.parse_args()

# ==========================
# CONFIGURATION
# ==========================
WEAPON_MODEL_PATH = args.weapon_model
MASK_MODEL_PATH = args.mask_model
CONF_THRESHOLD = args.conf
SOURCE = args.source
RECORD = args.record
RECORD_NAME = args.output
RESOLUTION = (640, 480)

# ==========================
# VERIFY MODEL PATHS
# ==========================
if not os.path.exists(WEAPON_MODEL_PATH):
    raise FileNotFoundError(f"[ERROR] Weapon model not found at {WEAPON_MODEL_PATH}.")
if not os.path.exists(MASK_MODEL_PATH):
    raise FileNotFoundError(f"[ERROR] Mask model not found at {MASK_MODEL_PATH}.")

print(f"[INFO] Loading Weapon Model: {WEAPON_MODEL_PATH}")
weapon_model = YOLO(WEAPON_MODEL_PATH)
weapon_labels = weapon_model.names
weapon_model.fuse()
try:
    weapon_model.to('cuda')
    print("[INFO] Weapon model on CUDA")
except:
    print("[INFO] Weapon model on CPU")

print(f"[INFO] Loading Mask Model: {MASK_MODEL_PATH}")
mask_model = YOLO(MASK_MODEL_PATH)
mask_labels = mask_model.names
mask_model.fuse()
try:
    mask_model.to('cuda')
    print("[INFO] Mask model on CUDA")
except:
    print("[INFO] Mask model on CPU")

# ==========================
# Arduino Setup
# ==========================
port = "COM9"
try:
    board = pyfirmata.Arduino(port, baudrate=57600)
    servo_pinX = board.get_pin('d:9:s')
    servo_pinY = board.get_pin('d:10:s')
    print("[INFO] Arduino connected successfully!")
except:
    print(f"[ERROR] Unable to connect to Arduino on {port}. Check connection.")
    board = None
    servo_pinX = None
    servo_pinY = None

servoPos = [90, 90]

# ==========================
# SOURCE TYPE
# ==========================
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
video_extensions = ['.mp4', '.avi', '.mov', '.mkv']

if SOURCE == "0":
    SOURCE = 0
    source_type = 'webcam'
elif SOURCE.startswith("usb"):
    SOURCE = int(SOURCE[3:])
    source_type = 'webcam'
elif os.path.isdir(SOURCE):
    source_type = 'folder'
elif os.path.isfile(SOURCE):
    ext = os.path.splitext(SOURCE)[1].lower()
    source_type = 'video' if ext in video_extensions else 'image'
else:
    raise ValueError("[ERROR] Invalid SOURCE.")

# ==========================
# SETUP VIDEO CAPTURE
# ==========================
if source_type in ['video', 'webcam']:
    cap = cv2.VideoCapture(SOURCE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
    cap.set(cv2.CAP_PROP_FPS, 60)

    if RECORD:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(RECORD_NAME, fourcc, 60, RESOLUTION)
else:
    cap = None

# ==========================
# COLORS
# ==========================
weapon_colors = {'gun': (0, 0, 255), 'knife': (0, 165, 255)}  # Gun=red, Knife=orange
face_colors = {'threat': (0, 0, 255), 'safe': (0, 255, 0)}    # Threat=red, Safe=green
fps_buffer = []
fps_avg_len = 30
ws, hs = RESOLUTION

# ==========================
# PROCESS FRAME FUNCTION
# ==========================
def process_frame(frame):
    # ========== Weapon Detection ==========
    weapon_results = weapon_model(frame, verbose=False, imgsz=640)[0]
    detections = weapon_results.boxes
    weapon_count = {'gun': 0, 'knife': 0}

    for box in detections:
        cls_id = int(box.cls.item())
        label = weapon_labels[cls_id].lower()
        conf = float(box.conf.item())
        if label not in ['gun', 'knife'] or conf < CONF_THRESHOLD:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy().squeeze())
        color = weapon_colors[label]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame, f"{label.upper()} {int(conf*100)}%", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        weapon_count[label] += 1

    # ========== Mask Detection ==========
    mask_results = mask_model(frame, verbose=False, imgsz=640)[0]
    mask_boxes = mask_results.boxes
    threat_count = 0
    safe_count = 0

    for box in mask_boxes:
        cls_id = int(box.cls.item())
        label = mask_labels[cls_id].lower()
        conf = float(box.conf.item())
        if conf < CONF_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy().squeeze())

        # Assuming your mask model labels:
        # 'mask' = threat (covered face), 'no_mask' = safe (uncovered)
        if 'mask' in label or 'cover' in label:
            color = face_colors['threat']
            text = "THREAT"
            threat_count += 1
        else:
            color = face_colors['safe']
            text = "SAFE"
            safe_count += 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame, f"{text} {int(conf*100)}%", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Servo Tracking (optional — track first face)
        fx, fy = (x1 + x2) // 2, (y1 + y2) // 2
        servoX = np.interp(fx, [0, ws], [180, 0])
        servoY = np.interp(fy, [0, hs], [180, 0])
        servoPos[0], servoPos[1] = servoX, servoY
        if servo_pinX and servo_pinY:
            servo_pinX.write(servoX)
            servo_pinY.write(servoY)

    return frame, weapon_count, threat_count, safe_count

# ==========================
# MAIN LOOP
# ==========================
if source_type in ['video', 'webcam']:
    print("[INFO] Starting video stream...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of stream or error reading video.")
            break

        start_time = time.perf_counter()
        frame, weapon_counts, threat_count, safe_count = process_frame(frame)
        end_time = time.perf_counter()

        fps = 1 / (end_time - start_time)
        fps_buffer.append(fps)
        if len(fps_buffer) > fps_avg_len:
            fps_buffer.pop(0)
        avg_fps = np.mean(fps_buffer)

        cv2.putText(frame, f"FPS: {avg_fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.putText(frame, f"Guns: {weapon_counts['gun']} Knives: {weapon_counts['knife']}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.putText(frame, f"THREAT: {threat_count}  SAFE: {safe_count}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.putText(frame, f'Servo X: {int(servoPos[0])} Y: {int(servoPos[1])}', (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow("YOLO - Weapon & Mask Threat Detection", frame)
        if RECORD:
            out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if RECORD:
        out.release()
    cv2.destroyAllWindows()

elif source_type == 'folder':
    print("[INFO] Processing images from folder...")
    for img_name in os.listdir(SOURCE):
        if os.path.splitext(img_name)[1].lower() not in image_extensions:
            continue
        img_path = os.path.join(SOURCE, img_name)
        frame = cv2.imread(img_path)
        frame, weapon_counts, threat_count, safe_count = process_frame(frame)
        cv2.putText(frame, f"Guns: {weapon_counts['gun']} Knives: {weapon_counts['knife']}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"THREAT: {threat_count}  SAFE: {safe_count}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("YOLO - Weapon & Mask Threat Detection", frame)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

elif source_type == 'image':
    print(f"[INFO] Processing single image: {SOURCE}")
    frame = cv2.imread(SOURCE)
    frame, weapon_counts, threat_count, safe_count = process_frame(frame)
    cv2.putText(frame, f"Guns: {weapon_counts['gun']} Knives: {weapon_counts['knife']}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"THREAT: {threat_count}  SAFE: {safe_count}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imshow("YOLO - Weapon & Mask Threat Detection", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
