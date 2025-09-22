import cv2
import os
import time
import numpy as np
from ultralytics import YOLO

# ==========================
# CONFIGURATION
# ==========================

MODEL_PATH = "runs/detect/train/weights/best.pt"  # Path to your trained YOLO model
SOURCE = 0  # 0 = Webcam, 'video.mp4' = Video file, 'images/' = Folder of images
CONF_THRESHOLD = 0.5  # Confidence threshold
RESOLUTION = (640, 480)  # Width x Height
RECORD = False  # Set True to save video output
RECORD_NAME = "knife_gun_detection.avi"

# ==========================
# LOAD YOLO MODEL
# ==========================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

print("[INFO] Loading YOLO model...")
model = YOLO(MODEL_PATH)
labels = model.names  # Example: {0: 'gun', 1: 'knife'}

print(f"[INFO] Classes in model: {labels}")

# ==========================
# SETUP VIDEO OR IMAGE SOURCE
# ==========================
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
video_extensions = ['.mp4', '.avi', '.mov', '.mkv']

if isinstance(SOURCE, int):
    source_type = 'webcam'
elif os.path.isdir(str(SOURCE)):
    source_type = 'folder'
elif os.path.isfile(str(SOURCE)):
    ext = os.path.splitext(SOURCE)[1].lower()
    source_type = 'video' if ext in video_extensions else 'image'
else:
    raise ValueError("Invalid SOURCE. Must be webcam index, image folder, or video file path.")

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
            print("[INFO] End of stream or error.")
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
