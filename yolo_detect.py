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
parser = argparse.ArgumentParser(description="Knife, Gun & Face Detection with Tracking (YOLO + Arduino)")
parser.add_argument("--model", type=str, default="runs/detect/train/weights/best.pt",
                    help="Path to your trained YOLO model (best.pt)")
parser.add_argument("--source", type=str, default="0",
                    help="Source: 0 for webcam, video file path, or image folder path")
parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
parser.add_argument("--record", action="store_true", help="Enable video recording")
parser.add_argument("--output", type=str, default="knife_gun_detection.avi", help="Output video file name")
args = parser.parse_args()

# ==========================
# CONFIGURATION
# ==========================
MODEL_PATH = args.model
CONF_THRESHOLD = args.conf
SOURCE = args.source
RECORD = args.record
RECORD_NAME = args.output
RESOLUTION = (640, 480)  # Width x Height

# ==========================
# VERIFY MODEL PATH
# ==========================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"[ERROR] Model not found at {MODEL_PATH}. "
                            f"Please provide the correct path using --model argument.")

print(f"[INFO] Loading YOLO model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
labels = model.names  # Example: {0: 'gun', 1: 'knife'}
print(f"[INFO] Classes in model: {labels}")

# ==========================
# Arduino Setup for Servo
# ==========================
port = "COM9"  # Change this to your correct COM port
try:
    board = pyfirmata.Arduino(port)
    servo_pinX = board.get_pin('d:9:s')   # Servo X on pin 9
    servo_pinY = board.get_pin('d:10:s')  # Servo Y on pin 10
    print("[INFO] Arduino connected successfully!")
except:
    print("[ERROR] Unable to connect to Arduino on COM9. Check connection.")
    board = None
    servo_pinX = None
    servo_pinY = None

servoPos = [90, 90]  # Initial servo position

# ==========================
# DETERMINE SOURCE TYPE
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
    raise ValueError("[ERROR] Invalid SOURCE. Use webcam index (0), image folder path, or video file path.")

# ==========================
# SETUP VIDEO CAPTURE
# ==========================
if source_type in ['video', 'webcam']:
    cap = cv2.VideoCapture(SOURCE)
    cap.set(3, RESOLUTION[0])
    cap.set(4, RESOLUTION[1])

    if RECORD:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(RECORD_NAME, fourcc, 30, RESOLUTION)
else:
    cap = None

# ==========================
# COLORS & VARIABLES
# ==========================
colors = {'gun': (0, 0, 255), 'knife': (255, 0, 0)}  # Red = Gun, Blue = Knife
fps_buffer = []
fps_avg_len = 30
ws, hs = RESOLUTION  # Camera resolution

# ==========================
# FACE DETECTION (Haar Cascade)
# ==========================
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# ==========================
# PROCESS FRAME FUNCTION
# ==========================
def process_frame(frame):
    results = model(frame, verbose=False)[0]
    detections = results.boxes
    object_count = {'gun': 0, 'knife': 0}
    face_coords = None  # Will store the center of detected face

    # ---------- YOLO Object Detection ----------
    for box in detections:
        cls_id = int(box.cls.item())
        label = labels[cls_id].lower()
        conf = float(box.conf.item())

        if label not in ['gun', 'knife']:
            continue

        if conf < CONF_THRESHOLD:
            continue

        # Get bounding box
        x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy().squeeze())

        # Draw bounding box
        color = colors[label]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label}: {int(conf * 100)}%", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Count detections
        object_count[label] += 1

    # ---------- Face Detection for Tracking ----------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3)

    if len(faces) > 0:
        (x, y, w, h) = faces[0]  # Track the first detected face
        fx, fy = x + w // 2, y + h // 2
        face_coords = (fx, fy)

        # Draw rectangle and crosshair
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.line(frame, (0, fy), (ws, fy), (0, 0, 0), 2)
        cv2.line(frame, (fx, hs), (fx, 0), (0, 0, 0), 2)
        cv2.putText(frame, "TARGET LOCKED", (450, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        # Servo control mapping
        servoX = np.interp(fx, [0, ws], [180, 0])
        servoY = np.interp(fy, [0, hs], [180, 0])
        servoX = max(0, min(180, servoX))
        servoY = max(0, min(180, servoY))

        servoPos[0] = servoX
        servoPos[1] = servoY

        # Send positions to Arduino
        if servo_pinX and servo_pinY:
            servo_pinX.write(servoX)
            servo_pinY.write(servoY)

    else:
        cv2.putText(frame, "NO TARGET", (450, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    return frame, object_count, face_coords

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
        frame, counts, face_coords = process_frame(frame)
        end_time = time.perf_counter()

        # Calculate FPS
        fps = 1 / (end_time - start_time)
        fps_buffer.append(fps)
        if len(fps_buffer) > fps_avg_len:
            fps_buffer.pop(0)
        avg_fps = np.mean(fps_buffer)

        # Display FPS
        cv2.putText(frame, f"FPS: {avg_fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Display object counts
        cv2.putText(frame, f"Guns: {counts['gun']} Knives: {counts['knife']}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Display servo positions
        cv2.putText(frame, f'Servo X: {int(servoPos[0])} deg', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f'Servo Y: {int(servoPos[1])} deg', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Show window
        cv2.imshow("YOLO + Face Tracking", frame)

        if RECORD:
            out.write(frame)

        # Quit on 'q'
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
        frame, counts, _ = process_frame(frame)

        cv2.putText(frame, f"Guns: {counts['gun']} Knives: {counts['knife']}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("YOLO + Face Tracking", frame)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

elif source_type == 'image':
    print(f"[INFO] Processing single image: {SOURCE}")
    frame = cv2.imread(SOURCE)
    frame, counts, _ = process_frame(frame)

    cv2.putText(frame, f"Guns: {counts['gun']} Knives: {counts['knife']}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("YOLO + Face Tracking", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
