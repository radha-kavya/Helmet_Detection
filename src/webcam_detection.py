
import cv2
import time
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from model_transfer import create_transfer_model

# ==========================================
# 1. CONFIGURATION
# ==========================================

DEVICE = torch.device("cpu")

MODEL_PATH = "models/transfer_epoch_2.pth"

CONFIDENCE_THRESHOLD = 0.50

CLASS_NAMES = {
    1: "Helmet",
    2: "No Helmet"
}

# ==========================================
# 2. LOAD MODEL
# ==========================================

print("Loading Helmet Detection Model...")

model = create_transfer_model(num_classes=3)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(checkpoint)
model.to(DEVICE)
model.eval()

print("Model loaded successfully!")

# ==========================================
# 3. OPEN WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError(
        "Webcam not available. Check camera connection."
    )

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Webcam started!")
print("Press Q to quit.")

# ==========================================
# 4. REAL-TIME DETECTION
# ==========================================

previous_time = time.perf_counter()

while True:

    success, frame = cap.read()

    if not success:
        print("Could not read webcam frame.")
        break

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    pil_image = Image.fromarray(rgb_frame)

    image_tensor = to_tensor(pil_image).to(DEVICE)

    with torch.no_grad():
        prediction = model([image_tensor])[0]

    boxes = prediction["boxes"].cpu()
    labels = prediction["labels"].cpu()
    scores = prediction["scores"].cpu()

    # Draw predictions
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

    # Calculate FPS
    current_time = time.perf_counter()

    fps = 1 / (current_time - previous_time)

    previous_time = current_time

    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    cv2.imshow(
        "Real-Time Helmet Detection",
        frame
    )

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==========================================
# 5. RELEASE RESOURCES
# ==========================================

cap.release()
cv2.destroyAllWindows()

print("Webcam detection stopped.")
