
from pathlib import Path
from collections import Counter
import random
import shutil

# ==========================================
# 1. PATHS AND SETTINGS
# ==========================================

SOURCE = Path("dataset_extracted/construction-ppe")
DESTINATION = Path("prepared_dataset")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Original class ID -> New class ID
CLASS_MAPPING = {
    0: 0,  # helmet
    6: 1   # no_helmet
}

CLASS_NAMES = {
    0: "helmet",
    1: "no_helmet"
}

RANDOM_SEED = 42


# ==========================================
# 2. CREATE OUTPUT FOLDERS
# ==========================================

for split in ["train", "val", "test"]:

    (DESTINATION / "images" / split).mkdir(
        parents=True, exist_ok=True
    )

    (DESTINATION / "labels" / split).mkdir(
        parents=True, exist_ok=True
    )


# ==========================================
# 3. PREPARE IMAGE-LABEL PAIRS
# ==========================================

def get_image_pairs(split):

    image_dir = SOURCE / "images" / split
    label_dir = SOURCE / "labels" / split

    images = sorted([
        p for p in image_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ])

    pairs = []

    for image_path in images:

        label_path = label_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            raise FileNotFoundError(
                f"Missing label: {label_path}"
            )

        pairs.append((image_path, label_path))

    return pairs


# ==========================================
# 4. CONVERT LABELS
# ==========================================

def copy_pair(image_path, label_path, split):

    destination_image = (
        DESTINATION / "images" / split / image_path.name
    )

    destination_label = (
        DESTINATION / "labels" / split /
        f"{image_path.stem}.txt"
    )

    shutil.copy2(image_path, destination_image)

    new_labels = []
    class_counts = Counter()

    with open(label_path, "r") as file:

        for line in file:

            parts = line.strip().split()

            if not parts:
                continue

            original_id = int(parts[0])

            if original_id in CLASS_MAPPING:

                new_id = CLASS_MAPPING[original_id]

                new_line = " ".join(
                    [str(new_id)] + parts[1:]
                )

                new_labels.append(new_line)
                class_counts[new_id] += 1

    # Empty annotation file is valid for an image
    # containing neither target class.

    with open(destination_label, "w") as file:

        if new_labels:
            file.write("\n".join(new_labels) + "\n")

    return class_counts


# ==========================================
# 5. SPLIT ORIGINAL TRAINING DATA
# ==========================================

train_pairs = get_image_pairs("train")

random.seed(RANDOM_SEED)
random.shuffle(train_pairs)

test_size = int(len(train_pairs) * 0.20)

test_pairs = train_pairs[:test_size]
new_train_pairs = train_pairs[test_size:]


# ==========================================
# 6. COPY TRAINING AND TEST DATA
# ==========================================

for split, pairs in [
    ("train", new_train_pairs),
    ("test", test_pairs)
]:

    total_counts = Counter()

    for image_path, label_path in pairs:

        counts = copy_pair(
            image_path, label_path, split
        )

        total_counts.update(counts)

    print(f"\n{split.upper()} DATASET")
    print("Images:", len(pairs))
    print("Helmet objects:", total_counts[0])
    print("No-helmet objects:", total_counts[1])


# ==========================================
# 7. COPY EXISTING VALIDATION DATA
# ==========================================

val_pairs = get_image_pairs("val")

val_counts = Counter()

for image_path, label_path in val_pairs:

    counts = copy_pair(
        image_path, label_path, "val"
    )

    val_counts.update(counts)

print("\nVALIDATION DATASET")
print("Images:", len(val_pairs))
print("Helmet objects:", val_counts[0])
print("No-helmet objects:", val_counts[1])


# ==========================================
# 8. SUMMARY
# ==========================================

print("\nCLASS MAPPING")
print(CLASS_NAMES)

print("\nPREPARED DATASET CREATED SUCCESSFULLY!")

print("\nLocation:", DESTINATION.resolve())

print("\nFinal image counts:")
for split in ["train", "val", "test"]:

    image_dir = DESTINATION / "images" / split

    print(
        split,
        ":",
        len(list(image_dir.iterdir()))
    )
