
from pathlib import Path

import torch
from torch.optim import SGD
from tqdm import tqdm

from dataset import create_dataloaders
from model_scratch import create_scratch_model


# ==================================================
# 1. SETTINGS
# ==================================================

DEVICE = torch.device("cpu")

BATCH_SIZE = 2
IMAGE_SIZE = 320
LEARNING_RATE = 0.005
NUM_EPOCHS = 2

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


# ==================================================
# 2. LOAD DATA
# ==================================================

train_loader, val_loader, _ = create_dataloaders(
    root_dir="prepared_dataset",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)


# ==================================================
# 3. CREATE MODEL AND OPTIMIZER
# ==================================================

model = create_scratch_model(num_classes=3)
model.to(DEVICE)

optimizer = SGD(
    model.parameters(),
    lr=LEARNING_RATE,
    momentum=0.9,
    weight_decay=0.0005
)


# ==================================================
# 4. TRAINING AND VALIDATION
# ==================================================

for epoch in range(NUM_EPOCHS):

    # --------------------------
    # Training
    # --------------------------

    model.train()
    train_loss_total = 0.0

    train_progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} - Training"
    )

    for images, targets in train_progress:

        images = [image.to(DEVICE) for image in images]

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

        train_loss_total += total_loss.item()

        train_progress.set_postfix(
            loss=f"{total_loss.item():.4f}"
        )

    average_train_loss = train_loss_total / len(train_loader)


    # --------------------------
    # Validation loss
    # --------------------------

    # TorchVision detection models return loss dictionaries
    # when in training mode, even when gradients are disabled.
    model.train()
    val_loss_total = 0.0

    with torch.no_grad():

        for images, targets in tqdm(
            val_loader,
            desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} - Validation"
        ):

            images = [image.to(DEVICE) for image in images]

            targets = [
                {
                    "boxes": target["boxes"].to(DEVICE),
                    "labels": target["labels"].to(DEVICE)
                }
                for target in targets
            ]

            loss_dict = model(images, targets)
            val_loss_total += sum(loss_dict.values()).item()

    average_val_loss = val_loss_total / len(val_loader)


    # --------------------------
    # Save checkpoint
    # --------------------------

    checkpoint_path = MODEL_DIR / f"scratch_epoch_{epoch + 1}.pth"

    torch.save(model.state_dict(), checkpoint_path)

    print(f"\nEpoch {epoch + 1} complete")
    print(f"Average training loss: {average_train_loss:.4f}")
    print(f"Average validation loss: {average_val_loss:.4f}")
    print(f"Saved model: {checkpoint_path}")


print("\nFull scratch-model training completed!")

