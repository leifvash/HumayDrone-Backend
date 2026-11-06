"""
ResNet-50 deep feature extractor and multi-class classifier for rice discoloration symptoms.
Targets 4 classes from Capstone Section 3.2.3.9.1:
0: Healthy
1: Bacterial Leaf Blight (BLB)
2: Dead Heart
3: White Head
"""
try:
    import torch
    import torch.nn as nn
    from torchvision.models import resnet50, ResNet50_Weights
except ImportError:
    torch = None
    nn = None
    resnet50 = None

if nn is not None and resnet50 is not None:
    class RiceDiscolorationResNet50(nn.Module):
        def __init__(self, num_classes=4, pretrained=True):
            super().__init__()
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            self.base = resnet50(weights=weights)
            in_features = self.base.fc.in_features
            # Replace final classification head with 4-class output
            self.base.fc = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(in_features, num_classes)
            )

        def forward(self, x):
            return self.base(x)
else:
    class RiceDiscolorationResNet50:
        pass
