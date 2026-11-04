
import time
from pathlib import Path

import cv2
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from model_transfer import create_transfer_model

# Configuration
DEVICE = torch.device("cpu")

VIDEO_PATH = "test_video.mp4"
MODEL_PATH = "models/transfer_epoch_2.pth"
OUTPUT_PATH = "outputs/helmet_detection_video.mp4"

CONFIDENCE_THRESHOLD = 0.50

CLASS_NAMES = {
    1: "Helmet",
    2: "No Helmet"
}

# Load model
print("Loading Transfer Learning model...")

model = create_transfer_model(num_classes=3)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(checkpoint)
model.to(DEVICE)
model.eval()

print("Model loaded successfully!")

# Check video
if not Path(VIDEO_PATH).exists():
    raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("Could not open video.")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0:
    fps = 25.0

Path("outputs").mkdir(exist_ok=True)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (width, height)
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError("Could not create output video.")

# Process video
frame_count = 0
start_time = time.perf_counter()

print("Starting video detection...")

while True:

    success, frame = cap.read()

    if not success:
        break

    frame_count += 1

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    pil_image = Image.fromarray(rgb_frame)

    image_tensor = to_tensor(pil_image).to(DEVICE)

    with torch.no_grad():
        prediction = model([image_tensor])[0]

    boxes = prediction["boxes"].cpu()
    labels = prediction["labels"].cpu()
    scores = prediction["scores"].cpu()

    for box, label, score in zip(boxes, labels, scores):

        confidence = score.item()

        if confidence < CONFIDENCE_THRESHOLD:
            continue

        class_id = label.item()

        if class_id not in CLASS_NAMES:
            continue

        x1, y1, x2, y2 = map(int, box.tolist())

        class_name = CLASS_NAMES[class_id]

        if class_id == 1:
            color = (0, 200, 0)
        else:
            color = (0, 0, 255)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        text = f"{class_name}: {confidence:.2f}"

        cv2.putText(
            frame,
            text,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    writer.write(frame)

    if frame_count % 10 == 0:
        print("Processed frames:", frame_count)

# Release resources
cap.release()
writer.release()

elapsed_time = time.perf_counter() - start_time

processing_fps = frame_count / elapsed_time if elapsed_time > 0 else 0

print("\nVideo detection completed!")
print("Total frames:", frame_count)
print(f"Processing FPS: {processing_fps:.2f}")
print("Output saved:", OUTPUT_PATH)
