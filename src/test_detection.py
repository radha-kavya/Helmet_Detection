
import torch
import torchvision

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN
from torchvision.ops import nms

print("PyTorch version:", torch.__version__)
print("TorchVision version:", torchvision.__version__)

# Test Faster R-CNN import
print("\nFaster R-CNN import: SUCCESS")

# Test Non-Maximum Suppression
boxes = torch.tensor([
    [10, 10, 100, 100],
    [20, 20, 90, 90]
], dtype=torch.float32)

scores = torch.tensor([0.9, 0.8])

result = nms(boxes, scores, 0.5)

print("NMS test: SUCCESS")
print("Selected boxes:", result)

print("\nTorchVision detection components are working!")
