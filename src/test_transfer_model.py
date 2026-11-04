
import torch

from dataset import create_dataloaders
from model_transfer import create_transfer_model


# ==================================================
# 1. DEVICE
# ==================================================

device = torch.device("cpu")


# ==================================================
# 2. LOAD ONE BATCH
# ==================================================

train_loader, _, _ = create_dataloaders(
    root_dir="prepared_dataset",
    image_size=320,
    batch_size=2
)

images, targets = next(iter(train_loader))


# ==================================================
# 3. CREATE TRANSFER LEARNING MODEL
# ==================================================

model = create_transfer_model(num_classes=3)

model.to(device)
model.train()


# ==================================================
# 4. PREPARE INPUTS
# ==================================================

images = [
    image.to(device)
    for image in images
]

targets = [
    {
        "boxes": target["boxes"].to(device),
        "labels": target["labels"].to(device)
    }
    for target in targets
]


# ==================================================
# 5. TEST FORWARD PASS
# ==================================================

loss_dict = model(images, targets)

print("\nTransfer Learning Detection Losses:")

for name, value in loss_dict.items():
    print(f"{name}: {value.item():.4f}")

total_loss = sum(loss_dict.values())

print("\nTotal Loss:", total_loss.item())

assert torch.isfinite(total_loss)

print("\nTransfer Learning Model Test Successful!")
