import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNet50Baseline(nn.Module):
    """
    ResNet-50 Continuous Regression Baseline for Pediatric Bone Age Estimation.
    Replaces the 1,000-class ImageNet fully connected head with a single output node.
    """

    def __init__(self, pretrained: bool = True):
        super(ResNet50Baseline, self).__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = resnet50(weights=weights)

        # Replace classification head with a single continuous regression scalar
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns shape (batch_size,)
        return self.backbone(x).squeeze(-1)


if __name__ == "__main__":
    # Sanity check
    model = ResNet50Baseline(pretrained=False)
    dummy_input = torch.randn(2, 3, 512, 512)
    output = model(dummy_input)
    print(f"ResNet-50 Baseline initialized successfully. Output shape: {output.shape}")