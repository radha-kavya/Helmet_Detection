
import torch

from dataset import create_dataloaders
from model_scratch import create_scratch_model


# Use CPU
device = torch.device("cpu")

# Load one batch
train_loader, _, _ = create_dataloaders(
    root_dir="prepared_dataset",
    image_size=320,
    batch_size=2
)

images, targets = next(iter(train_loader))

# Create the scratch model
model = create_scratch_model(num_classes=3)
model.to(device)
model.train()

# Move images and targets to CPU
images = [image.to(device) for image in images]

targets = [
    {
        "boxes": target["boxes"].to(device),
        "labels": target["labels"].to(device)
    }
    for target in targets
]

# Run one forward pass and calculate detection losses
loss_dict = model(images, targets)

print("\nDetection losses:")

for loss_name, loss_value in loss_dict.items():
    print(f"{loss_name}: {loss_value.item():.4f}")

total_loss = sum(loss_dict.values())

print("\nTotal loss:", total_loss.item())

assert torch.isfinite(total_loss), "Loss contains NaN or infinity"

print("\nScratch Faster R-CNN model test successful!")
