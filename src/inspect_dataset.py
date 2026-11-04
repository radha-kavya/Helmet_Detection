
from pathlib import Path
from collections import Counter
from PIL import Image

# Dataset paths
DATASET = Path("dataset_extracted/construction-ppe")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for split in ["train", "val"]:

    image_dir = DATASET / "images" / split
    label_dir = DATASET / "labels" / split

    images = [
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    labels = list(label_dir.glob("*.txt"))

    image_names = {p.stem for p in images}
    label_names = {p.stem for p in labels}

    missing_labels = image_names - label_names
    missing_images = label_names - image_names

    class_counts = Counter()
    invalid_rows = 0
    invalid_coordinates = 0

    for label_file in labels:
        with open(label_file, "r") as file:
            for line in file:
                parts = line.strip().split()

                if not parts:
                    continue

                if len(parts) != 5:
                    invalid_rows += 1
                    continue

                try:
                    class_id = int(parts[0])
                    coordinates = list(map(float, parts[1:]))

                    if class_id < 0 or class_id > 11:
                        invalid_rows += 1
                        continue

                    if not all(0 <= value <= 1 for value in coordinates):
                        invalid_coordinates += 1
                        continue

                    class_counts[class_id] += 1

                except ValueError:
                    invalid_rows += 1

    print("\n" + "=" * 45)
    print(f"{split.upper()} DATASET")
    print("=" * 45)

    print("Images:", len(images))
    print("Label files:", len(labels))
    print("Images without labels:", len(missing_labels))
    print("Labels without images:", len(missing_images))
    print("Invalid annotation rows:", invalid_rows)
    print("Invalid coordinates:", invalid_coordinates)

    print("\nClass distribution:")

    for class_id, count in sorted(class_counts.items()):
        print(f"Class {class_id}: {count}")

    print("\nImage extensions:")

    extensions = Counter(p.suffix.lower() for p in images)
    print(dict(extensions))

print("\nDataset inspection completed!")
