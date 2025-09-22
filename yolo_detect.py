import cv2
import os
import time
import numpy as np
import argparse
from ultralytics import YOLO

# ==========================
# ARGUMENT PARSER
# ==========================
parser = argparse.ArgumentParser(description="Knife & Gun Detection with YOLO")
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
# DETERMINE SOURCE TYPE
# ==========================
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
video_extensions = ['.mp4', '.avi', '.mov', '.mkv']

# If user enters '0', use webcam
if SOURCE == "0":
    SOURCE = 0
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

# ==========================
# PROCESS FRAME FUNCTION
# ==========================
def process_frame(frame):
    results = model(frame, verbose=False)[0]
    detections = results.boxes
    object_count = {'gun': 0, 'knife': 0}

    for box in detections:
        cls_id = int(box.cls.item())
        label = labels[cls_id].lower()
        conf = float(box.conf.item())

        # Only detect gun or knife
        if label not in ['gun', 'knife']:
            continue

        if conf < CONF_THRESHOLD:
            continue

        # Get bounding box
        x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy().squeeze())

        # Draw box and label
        color = colors[label]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label}: {int(conf * 100)}%", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Count detections
        object_count[label] += 1

    return frame, object_count

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
        frame, counts = process_frame(frame)
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

        # Show window
        cv2.imshow("Knife & Gun Detection", frame)

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
        frame, counts = process_frame(frame)

        cv2.putText(frame, f"Guns: {counts['gun']} Knives: {counts['knife']}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Knife & Gun Detection", frame)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

elif source_type == 'image':
    print(f"[INFO] Processing single image: {SOURCE}")
    frame = cv2.imread(SOURCE)
    frame, counts = process_frame(frame)

    cv2.putText(frame, f"Guns: {counts['gun']} Knives: {counts['knife']}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Knife & Gun Detection", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
