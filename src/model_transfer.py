
import torch.nn as nn

from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights
)

from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor
)


def create_transfer_model(num_classes=3):

    # Load Faster R-CNN with pretrained ResNet50-FPN weights
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT

    model = fasterrcnn_resnet50_fpn(
        weights=weights,
        min_size=320,
        max_size=320
    )

    # Replace the pretrained classification head
    # with a new head for our 2 object classes + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        num_classes
    )

    # Freeze early ResNet layers
    # Fine-tune the final ResNet layer
    for name, parameter in model.backbone.body.named_parameters():

        parameter.requires_grad = name.startswith("layer4.")

    # Keep the FPN, RPN and detection head trainable
    for parameter in model.backbone.fpn.parameters():
        parameter.requires_grad = True

    return model
