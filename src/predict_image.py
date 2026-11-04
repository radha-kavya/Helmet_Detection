
import torch
import cv2
from pathlib import Path
from PIL import Image
from torchvision.transforms.functional import to_tensor

from model_transfer import create_transfer_model

# ==========================================
# 1. CONFIGURATION
# ==========================================

DEVICE = torch.device("cpu")

IMAGE_PATH = "test_image.jpg"
MODEL_PATH = "models/transfer_epoch_2.pth"

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

CONFIDENCE_THRESHOLD = 0.50

CLASS_NAMES = {
    1: "Helmet",
    2: "No Helmet"
}

# ==========================================
# 2. LOAD MODEL
# ==========================================

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

# ==========================================
# 3. LOAD IMAGE
# ==========================================

if not Path(IMAGE_PATH).exists():
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

image = Image.open(IMAGE_PATH).convert("RGB")

image_tensor = to_tensor(image).to(DEVICE)

print("Image loaded:", image.size)

# ==========================================
# 4. MAKE PREDICTIONS
# ==========================================

print("Detecting objects...")

with torch.no_grad():
    prediction = model([image_tensor])[0]

boxes = prediction["boxes"].cpu()
labels = prediction["labels"].cpu()
scores = prediction["scores"].cpu()

# ==========================================
# 5. DRAW BOUNDING BOXES
# ==========================================

image_cv = cv2.cvtColor(
    cv2.imread(IMAGE_PATH),
    cv2.COLOR_BGR2RGB
)

detection_count = 0

for box, label, score in zip(boxes, labels, scores):

    confidence = score.item()

    if confidence < CONFIDENCE_THRESHOLD:
        continue

    class_id = label.item()

    if class_id not in CLASS_NAMES:
        continue

    x1, y1, x2, y2 = map(int, box.tolist())

    class_name = CLASS_NAMES[class_id]

    color = (0, 200, 0) if class_id == 1 else (255, 0, 0)

    # Draw bounding box
    cv2.rectangle(
        image_cv,
        (x1, y1),
        (x2, y2),
        color,
        2
    )

    # Draw label and confidence
    text = f"{class_name}: {confidence:.2f}"

    cv2.putText(
        image_cv,
        text,
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2
    )

    detection_count += 1

# ==========================================
# 6. SAVE RESULT
# ==========================================

output_path = OUTPUT_DIR / "helmet_detection_result.jpg"

cv2.imwrite(
    str(output_path),
    cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
)

print("\nDetection completed!")
print("Total detections:", detection_count)
print("Saved result:", output_path)

# Display result
cv2.imshow(
    "Helmet Detection - Transfer Learning",
    cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
)

cv2.waitKey(0)
cv2.destroyAllWindows()
