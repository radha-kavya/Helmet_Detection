
import torch
import torch.nn as nn

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign


# ==================================================
# 1. CUSTOM CNN BACKBONE
# ==================================================

class CustomCNNBackbone(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            # Input: 3 x 320 x 320
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 160 x 160 -> 80 x 80
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 80 x 80 -> 40 x 40
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 40 x 40 -> 20 x 20
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Faster R-CNN needs to know the backbone output channels
        self.out_channels = 256

    def forward(self, images):
        return self.features(images)


# ==================================================
# 2. CREATE FASTER R-CNN MODEL
# ==================================================

def create_scratch_model(num_classes=3):

    # Custom CNN with randomly initialized weights
    backbone = CustomCNNBackbone()

    # Anchor boxes used by the Region Proposal Network
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    # Pool features from the backbone's single feature map
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=["0"],
        output_size=7,
        sampling_ratio=2
    )

    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        min_size=320,
        max_size=320
    )

    return model
