
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from dataset import create_dataloaders
from model_transfer import create_transfer_model


# ==========================================
# 1. CONFIGURATION
# ==========================================

DEVICE = torch.device("cpu")

EPOCHS = 2
IMAGE_SIZE = 320
BATCH_SIZE = 2

DATASET_PATH = "prepared_dataset"
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# ==========================================
# 2. LOAD DATA
# ==========================================

train_loader, val_loader, _ = create_dataloaders(
    root_dir=DATASET_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

print("Training images:", len(train_loader.dataset))
print("Validation images:", len(val_loader.dataset))


# ==========================================
# 3. CREATE TRANSFER LEARNING MODEL
# ==========================================

model = create_transfer_model(num_classes=3)
model.to(DEVICE)

# Only train parameters that are not frozen
trainable_parameters = [
    parameter
    for parameter in model.parameters()
    if parameter.requires_grad
]

optimizer = torch.optim.SGD(
    trainable_parameters,
    lr=0.001,
    momentum=0.9,
    weight_decay=0.0005
)

print("\nTransfer learning model created!")
print("Trainable parameters:", sum(
    parameter.numel() for parameter in trainable_parameters
))


# ==========================================
# 4. TRAINING HISTORY
# ==========================================

history = {
    "train_loss": [],
    "val_loss": []
}

start_time = time.time()


# ==========================================
# 5. TRAINING LOOP
# ==========================================

for epoch in range(EPOCHS):

    model.train()

    total_train_loss = 0.0

    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS} - Training"
    )

    for images, targets in progress_bar:

        images = [
            image.to(DEVICE)
            for image in images
        ]

        targets = [
            {
                "boxes": target["boxes"].to(DEVICE),
                "labels": target["labels"].to(DEVICE)
            }
            for target in targets
        ]

        optimizer.zero_grad()

        loss_dict = model(images, targets)

        total_loss = sum(loss_dict.values())

        total_loss.backward()

        optimizer.step()

        total_train_loss += total_loss.item()

        progress_bar.set_postfix(
            loss=f"{total_loss.item():.4f}"
        )

    average_train_loss = (
        total_train_loss / len(train_loader)
    )


    # ======================================
    # 6. VALIDATION LOOP
    # ======================================

    # TorchVision Faster R-CNN returns loss
    # dictionaries when the model is in train mode.
    model.train()

    # Keep BatchNorm running statistics unchanged
    # during validation.
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()

    total_val_loss = 0.0

    with torch.no_grad():

        progress_bar = tqdm(
            val_loader,
            desc=f"Epoch {epoch + 1}/{EPOCHS} - Validation"
        )

        for images, targets in progress_bar:

            images = [
                image.to(DEVICE)
                for image in images
            ]

            targets = [
                {
                    "boxes": target["boxes"].to(DEVICE),
                    "labels": target["labels"].to(DEVICE)
                }
                for target in targets
            ]

            loss_dict = model(images, targets)

            total_loss = sum(loss_dict.values())

            total_val_loss += total_loss.item()

            progress_bar.set_postfix(
                loss=f"{total_loss.item():.4f}"
            )

    average_val_loss = (
        total_val_loss / len(val_loader)
    )


    # ======================================
    # 7. SAVE HISTORY
    # ======================================

    history["train_loss"].append(average_train_loss)
    history["val_loss"].append(average_val_loss)

    print(f"\nEpoch {epoch + 1}/{EPOCHS} completed")

    print(f"Training Loss: {average_train_loss:.4f}")
    print(f"Validation Loss: {average_val_loss:.4f}")


    # ======================================
    # 8. SAVE MODEL CHECKPOINT
    # ======================================

    checkpoint_path = MODEL_DIR / f"transfer_epoch_{epoch + 1}.pth"

    torch.save(model.state_dict(), checkpoint_path)

    print("Saved model:", checkpoint_path)


    with open(OUTPUT_DIR / "transfer_history.json", "w") as file:
        json.dump(history, file, indent=4)


# ==========================================
# 9. TRAINING TIME
# ==========================================

end_time = time.time()

training_time = end_time - start_time

print("\n================================")
print("TRANSFER LEARNING TRAINING DONE")
print("================================")

print(f"Total training time: {training_time / 60:.2f} minutes")

print("\nTraining Loss History:", history["train_loss"])
print("Validation Loss History:", history["val_loss"])

print("\nSaved checkpoints:")
print("models/transfer_epoch_1.pth")
print("models/transfer_epoch_2.pth")

print("\nSaved training history:")
print("outputs/transfer_history.json")
