
import json
import time
from pathlib import Path

import torch
from torchvision.ops import box_iou

from dataset import create_dataloaders
from model_scratch import create_scratch_model
from model_transfer import create_transfer_model


# ==========================================
# 1. CONFIGURATION
# ==========================================

DEVICE = torch.device("cpu")
IOU_THRESHOLD = 0.50
CONFIDENCE_THRESHOLD = 0.50

MODEL_PATHS = {
    "Scratch CNN": "models/scratch_epoch_2.pth",
    "Transfer Learning": "models/transfer_epoch_2.pth"
}

CLASS_NAMES = {
    1: "helmet",
    2: "no_helmet"
}

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ==========================================
# 2. LOAD TEST DATA
# ==========================================

_, _, test_loader = create_dataloaders(
    root_dir="prepared_dataset",
    image_size=320,
    batch_size=1
)

print("Test images:", len(test_loader.dataset))


# ==========================================
# 3. CREATE MODEL
# ==========================================

def load_model(model_name, checkpoint_path):

    if model_name == "Scratch CNN":
        model = create_scratch_model(num_classes=3)
    else:
        model = create_transfer_model(num_classes=3)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )

    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()

    return model


# ==========================================
# 4. CALCULATE IoU AND MATCH BOXES
# ==========================================

def calculate_metrics(model):

    total_tp = 0
    total_fp = 0
    total_fn = 0

    matched_ious = []

    class_results = {
        1: {"tp": 0, "fp": 0, "fn": 0},
        2: {"tp": 0, "fp": 0, "fn": 0}
    }

    inference_times = []

    for images, targets in test_loader:

        image = images[0].to(DEVICE)
        target = targets[0]

        # Measure model inference time
        start = time.perf_counter()

        with torch.no_grad():
            prediction = model([image])[0]

        elapsed = time.perf_counter() - start
        inference_times.append(elapsed)

        pred_boxes = prediction["boxes"].cpu()
        pred_labels = prediction["labels"].cpu()
        pred_scores = prediction["scores"].cpu()

        gt_boxes = target["boxes"].cpu()
        gt_labels = target["labels"].cpu()

        # Remove low-confidence predictions
        keep = pred_scores >= CONFIDENCE_THRESHOLD

        pred_boxes = pred_boxes[keep]
        pred_labels = pred_labels[keep]
        pred_scores = pred_scores[keep]

        # Process each class separately
        for class_id in [1, 2]:

            pred_indices = torch.where(
                pred_labels == class_id
            )[0]

            gt_indices = torch.where(
                gt_labels == class_id
            )[0]

            class_pred_boxes = pred_boxes[pred_indices]
            class_gt_boxes = gt_boxes[gt_indices]

            matched_gt = set()

            # Highest confidence predictions first
            if len(pred_indices) > 0:

                order = torch.argsort(
                    pred_scores[pred_indices],
                    descending=True
                )

                class_pred_boxes = class_pred_boxes[order]

            # Calculate IoU between predicted and actual boxes
            if len(class_pred_boxes) > 0 and len(class_gt_boxes) > 0:

                iou_matrix = box_iou(
                    class_pred_boxes,
                    class_gt_boxes
                )

            else:
                iou_matrix = torch.zeros(
                    (len(class_pred_boxes), len(class_gt_boxes))
                )

            # Match predictions to ground-truth boxes
            for pred_index in range(len(class_pred_boxes)):

                if len(class_gt_boxes) == 0:
                    class_results[class_id]["fp"] += 1
                    total_fp += 1
                    continue

                ious = iou_matrix[pred_index].clone()

                # Ignore already matched ground-truth boxes
                for gt_index in matched_gt:
                    ious[gt_index] = -1

                best_iou, best_gt = torch.max(ious, dim=0)

                if best_iou.item() >= IOU_THRESHOLD:

                    matched_gt.add(best_gt.item())

                    class_results[class_id]["tp"] += 1
                    total_tp += 1

                    matched_ious.append(best_iou.item())

                else:

                    class_results[class_id]["fp"] += 1
                    total_fp += 1

            # Unmatched actual objects are false negatives
            missed = len(class_gt_boxes) - len(matched_gt)

            class_results[class_id]["fn"] += missed
            total_fn += missed

    # ======================================
    # 5. OVERALL METRICS
    # ======================================

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0

    mean_iou = (
        sum(matched_ious) / len(matched_ious)
        if matched_ious else 0
    )

    average_inference_time = (
        sum(inference_times) / len(inference_times)
    )

    fps = (
        1 / average_inference_time
        if average_inference_time > 0 else 0
    )

    return {
        "precision": precision,
        "recall": recall,
        "mean_iou": mean_iou,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "average_inference_seconds": average_inference_time,
        "average_inference_ms": average_inference_time * 1000,
        "fps": fps,
        "class_results": class_results
    }


# ==========================================
# 6. EVALUATE BOTH MODELS
# ==========================================

all_results = {}

for model_name, checkpoint_path in MODEL_PATHS.items():

    print("\n" + "=" * 50)
    print("Evaluating:", model_name)
    print("=" * 50)

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    model = load_model(model_name, checkpoint_path)

    results = calculate_metrics(model)

    model_size_mb = (
        Path(checkpoint_path).stat().st_size / (1024 ** 2)
    )

    results["model_size_mb"] = model_size_mb

    all_results[model_name] = results

    print(f"\nPrecision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"Mean IoU: {results['mean_iou']:.4f}")

    print("\nTrue Positives:", results["true_positives"])
    print("False Positives:", results["false_positives"])
    print("False Negatives:", results["false_negatives"])

    print(
        f"\nAverage inference time: "
        f"{results['average_inference_ms']:.2f} ms"
    )

    print(f"FPS: {results['fps']:.2f}")
    print(f"Model size: {model_size_mb:.2f} MB")

    print("\nClass-wise results:")

    for class_id, values in results["class_results"].items():

        name = CLASS_NAMES[class_id]

        tp = values["tp"]
        fp = values["fp"]
        fn = values["fn"]

        class_precision = tp / (tp + fp) if (tp + fp) else 0
        class_recall = tp / (tp + fn) if (tp + fn) else 0

        print(
            f"{name}: "
            f"Precision={class_precision:.4f}, "
            f"Recall={class_recall:.4f}"
        )

    del model


# ==========================================
# 7. SAVE RESULTS
# ==========================================

output_path = OUTPUT_DIR / "evaluation_results.json"

with open(output_path, "w") as file:
    json.dump(all_results, file, indent=4)

print("\n========================================")
print("BOTH MODELS EVALUATED SUCCESSFULLY!")
print("========================================")
print("Results saved to:", output_path)
