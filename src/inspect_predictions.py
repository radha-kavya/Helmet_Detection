
import torch
from pathlib import Path

from dataset import create_dataloaders
from model_scratch import create_scratch_model
from model_transfer import create_transfer_model

DEVICE = torch.device("cpu")

_, _, test_loader = create_dataloaders(
    root_dir="prepared_dataset",
    image_size=320,
    batch_size=1
)

def inspect_model(name, model_path, model_type):

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    if model_type == "scratch":
        model = create_scratch_model(num_classes=3)
    else:
        model = create_transfer_model(num_classes=3)

    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.eval()

    thresholds = [0.05, 0.10, 0.25, 0.50]
    counts = {threshold: 0 for threshold in thresholds}
    max_scores = []

    with torch.no_grad():

        for index, (images, targets) in enumerate(test_loader):

            if index >= 20:
                break

            prediction = model([images[0]])[0]

            scores = prediction["scores"]

            if len(scores) > 0:

                max_scores.append(scores.max().item())

                for threshold in thresholds:
                    counts[threshold] += (
                        scores >= threshold
                    ).sum().item()

    print("\nImages inspected:", len(max_scores))

    if max_scores:
        print("Average maximum confidence:",
              sum(max_scores) / len(max_scores))

        print("Highest confidence:",
              max(max_scores))

    print("\nPredictions at different thresholds:")

    for threshold, count in counts.items():
        print(f"Threshold {threshold:.2f}: {count} predictions")

    del model


inspect_model(
    "Scratch CNN",
    "models/scratch_epoch_2.pth",
    "scratch"
)

inspect_model(
    "Transfer Learning",
    "models/transfer_epoch_2.pth",
    "transfer"
)
