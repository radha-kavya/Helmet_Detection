
import json
from pathlib import Path

import matplotlib.pyplot as plt

# Scratch CNN training history
scratch_train = [0.2605, 0.1760]
scratch_val = [0.1811, 0.1755]

# Load Transfer Learning history
with open("outputs/transfer_history.json", "r") as file:
    transfer_history = json.load(file)

transfer_train = transfer_history["train_loss"]
transfer_val = transfer_history["val_loss"]

epochs = [1, 2]

# Create two comparison plots
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Scratch CNN
axes[0].plot(
    epochs, scratch_train,
    marker="o", label="Training Loss"
)
axes[0].plot(
    epochs, scratch_val,
    marker="s", label="Validation Loss"
)
axes[0].set_title("Scratch CNN")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_xticks(epochs)
axes[0].legend()
axes[0].grid(True)

# Transfer Learning
axes[1].plot(
    epochs, transfer_train,
    marker="o", label="Training Loss"
)
axes[1].plot(
    epochs, transfer_val,
    marker="s", label="Validation Loss"
)
axes[1].set_title("Transfer Learning")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].set_xticks(epochs)
axes[1].legend()
axes[1].grid(True)

plt.suptitle("Helmet Detection - Training and Validation Loss")
plt.tight_layout()

Path("outputs").mkdir(exist_ok=True)

plt.savefig(
    "outputs/training_loss_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Loss comparison graph saved successfully!")
print("Location: outputs/training_loss_comparison.png")
