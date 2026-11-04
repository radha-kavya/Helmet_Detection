
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF


class HelmetDataset(Dataset):

    def __init__(self, root_dir, split="train", image_size=320):

        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size

        self.image_dir = self.root_dir / "images" / split
        self.label_dir = self.root_dir / "labels" / split

        self.image_paths = sorted(self.image_dir.glob("*.jpg"))

        self.class_names = {
            0: "helmet",
            1: "no_helmet"
        }

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        image_path = self.image_paths[index]

        label_path = (
            self.label_dir / f"{image_path.stem}.txt"
        )

        # Load image
        image = Image.open(image_path).convert("RGB")

        original_width, original_height = image.size

        boxes = []
        labels = []

        # Read annotations
        with open(label_path, "r") as file:

            for line in file:

                parts = line.strip().split()

                if len(parts) != 5:
                    continue

                class_id = int(parts[0])

                x_center, y_center, box_width, box_height = (
                    map(float, parts[1:])
                )

                # Convert normalized YOLO coordinates
                # to absolute corner coordinates

                x_center *= original_width
                y_center *= original_height
                box_width *= original_width
                box_height *= original_height

                x_min = x_center - box_width / 2
                y_min = y_center - box_height / 2

                x_max = x_center + box_width / 2
                y_max = y_center + box_height / 2

                boxes.append([
                    x_min,
                    y_min,
                    x_max,
                    y_max
                ])

                # Torchvision detection models expect
                # background = 0 and object classes from 1
                labels.append(class_id + 1)

        # Resize image
        image = TF.resize(
            image,
            [self.image_size, self.image_size]
        )

        # Convert to tensor and normalize to [0, 1]
        image = TF.to_tensor(image)

        # Scale bounding boxes to resized image
        scale_x = self.image_size / original_width
        scale_y = self.image_size / original_height

        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)

            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y

            # Clip boxes to image boundaries
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(
                0, self.image_size
            )

            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(
                0, self.image_size
            )

        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([index])
        }

        return image, target


# ==========================================
# COLLATE FUNCTION
# ==========================================

def detection_collate(batch):

    return tuple(zip(*batch))


# ==========================================
# DATALOADER CREATION
# ==========================================

def create_dataloaders(
    root_dir="prepared_dataset",
    image_size=320,
    batch_size=2
):

    train_dataset = HelmetDataset(
        root_dir,
        "train",
        image_size
    )

    val_dataset = HelmetDataset(
        root_dir,
        "val",
        image_size
    )

    test_dataset = HelmetDataset(
        root_dir,
        "test",
        image_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=detection_collate
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=detection_collate
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=detection_collate
    )

    return train_loader, val_loader, test_loader
