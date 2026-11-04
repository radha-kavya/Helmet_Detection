
import torch

from dataset import create_dataloaders


# ==================================================
# 1. CREATE DATALOADERS
# ==================================================

train_loader, val_loader, test_loader = create_dataloaders(
    root_dir="prepared_dataset",
    image_size=320,
    batch_size=2
)


# ==================================================
# 2. GET ONE TRAINING BATCH
# ==================================================

images, targets = next(iter(train_loader))


# ==================================================
# 3. IMAGE INFORMATION
# ==================================================

print("\n========== IMAGE INFORMATION ==========")

print("Number of images:", len(images))
print("Image shape:", images[0].shape)
print("Image data type:", images[0].dtype)


# ==================================================
# 4. BOUNDING BOX INFORMATION
# ==================================================

print("\n========== BOUNDING BOX INFORMATION ==========")

print("Bounding boxes:")
print(targets[0]["boxes"])

print("Bounding box shape:", targets[0]["boxes"].shape)


# ==================================================
# 5. CLASS LABEL INFORMATION
# ==================================================

print("\n========== CLASS LABEL INFORMATION ==========")

print("Class labels:")
print(targets[0]["labels"])

class_names = {
    1: "helmet",
    2: "no_helmet"
}

print("\nClass names:")

for label in targets[0]["labels"]:
    print(class_names[label.item()])


# ==================================================
# 6. DATALOADER INFORMATION
# ==================================================

print("\n========== DATALOADER INFORMATION ==========")

print("Training images:", len(train_loader.dataset))
print("Validation images:", len(val_loader.dataset))
print("Testing images:", len(test_loader.dataset))

print("Training batches:", len(train_loader))
print("Validation batches:", len(val_loader))
print("Testing batches:", len(test_loader))


# ==================================================
# 7. VALIDATE TENSOR DIMENSIONS
# ==================================================

assert len(images) == 2

assert images[0].shape == (3, 320, 320)

assert targets[0]["boxes"].ndim == 2

assert targets[0]["boxes"].shape[1] == 4

assert targets[0]["labels"].dtype == torch.int64

assert targets[0]["boxes"].dtype == torch.float32


# ==================================================
# 8. SUCCESS MESSAGE
# ==================================================

print("\n========================================")
print("DATALOADER WORKING SUCCESSFULLY!")
print("========================================")
