import cv2
import threading
import time
from ultralytics import YOLO

# ==================================
# CONFIGURATION
# ==================================

CAMERA_URL = "http://IPadress:8080/video"

MODEL_PATH = "yolov8s.pt"

IMG_SIZE = 640
SKIP_FRAMES = 3

# ==================================
# LOAD YOLO
# ==================================

print("Loading YOLO...")

model = YOLO(MODEL_PATH)

model.to("cuda")

print(
    "Device:",
    next(model.model.parameters()).device
)

print("YOLO Loaded")

# ==================================
# GLOBALS
# ==================================

latest_frame = None
running = True

# ==================================
# CAMERA THREAD
# ==================================

def camera_thread():

    global latest_frame

    cap = cv2.VideoCapture(
        CAMERA_URL,
        cv2.CAP_FFMPEG
    )

    cap.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1
    )

    while running:

        ret, frame = cap.read()

        if ret:
            latest_frame = frame

    cap.release()

# ==================================
# START CAMERA THREAD
# ==================================

threading.Thread(
    target=camera_thread,
    daemon=True
).start()

# ==================================
# WINDOW
# ==================================

cv2.namedWindow(
    "VisionGuard YOLO",
    cv2.WINDOW_NORMAL
)

frame_counter = 0
last_results = None

# ==================================
# MAIN LOOP
# ==================================

while True:

    if latest_frame is None:

        cv2.waitKey(1)
        continue

    frame = latest_frame.copy()

    frame = cv2.resize(
        frame,
        (640, 360)
    )

    frame_counter += 1

    # YOLO every N frames
    if frame_counter % SKIP_FRAMES == 0:

        start = time.time()

        results = model(
            frame,
            imgsz=IMG_SIZE,
            conf=0.15,
            verbose=False,
            device=0
        )

        inference_time = round(
            (time.time() - start) * 1000
        )

        print(
            f"Inference: {inference_time} ms"
        )

        last_results = results

    if last_results is not None:

        annotated = last_results[0].plot()

    else:

        annotated = frame

    cv2.imshow(
        "VisionGuard YOLO",
        annotated
    )

    if cv2.waitKey(1) == 27:

        running = False
        break

# ==================================
# SHUTDOWN
# ==================================

cv2.destroyAllWindows()
