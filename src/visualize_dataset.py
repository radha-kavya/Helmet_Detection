
from pathlib import Path
import random

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# ==========================================
# 1. SETTINGS
# ==========================================

DATASET = Path("prepared_dataset")

CLASS_NAMES = {
    0: "Helmet",
    1: "No Helmet"
}

SPLIT = "train"
NUMBER_OF_IMAGES = 4

random.seed(42)

image_dir = DATASET / "images" / SPLIT
label_dir = DATASET / "labels" / SPLIT

# ==========================================
# 2. GET RANDOM IMAGES
# ==========================================

images = list(image_dir.glob("*.jpg"))

selected_images = random.sample(
    images,
    min(NUMBER_OF_IMAGES, len(images))
)

# ==========================================
# 3. DISPLAY IMAGES AND BOXES
# ==========================================

fig, axes = plt.subplots(
    2, 2,
    figsize=(14, 10)
)

axes = axes.flatten()

for ax, image_path in zip(axes, selected_images):

    image = Image.open(image_path).convert("RGB")

    width, height = image.size

    ax.imshow(image)

    label_path = label_dir / f"{image_path.stem}.txt"

    box_count = 0

    if label_path.exists():

        with open(label_path, "r") as file:

            for line in file:

                parts = line.strip().split()

                if len(parts) != 5:
                    continue

                class_id = int(parts[0])

                x_center, y_center, box_width, box_height = (
                    map(float, parts[1:])
                )

                # YOLO normalized coordinates
                x_center *= width
                y_center *= height
                box_width *= width
                box_height *= height

                # Convert center coordinates to corners
                x_min = x_center - box_width / 2
                y_min = y_center - box_height / 2

                color = "lime" if class_id == 0 else "red"

                rectangle = patches.Rectangle(
                    (x_min, y_min),
                    box_width,
                    box_height,
                    linewidth=2,
                    edgecolor=color,
                    facecolor="none"
                )

                ax.add_patch(rectangle)

                ax.text(
                    x_min,
                    max(0, y_min - 5),
                    CLASS_NAMES[class_id],
                    color="white",
                    fontsize=9,
                    backgroundcolor=color
                )

                box_count += 1

    ax.set_title(
        f"{image_path.name}\nObjects: {box_count}"
    )

    ax.axis("off")

plt.tight_layout()

plt.show()
