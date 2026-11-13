"""
ResNet-50 deep feature extractor and multi-class classifier for rice discoloration symptoms.
Targets 3 active classes:
0: Healthy
1: Bacterial Leaf Blight (BLB)
2: White Head

Implements pure PyTorch residual blocks to eliminate torchvision dependencies and save 110MB RAM.
"""
try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

if nn is not None:
    class Bottleneck(nn.Module):
        expansion = 4

        def __init__(self, inplanes, planes, stride=1, downsample=None):
            super().__init__()
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(planes * self.expansion)
            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample
            self.stride = stride

        def forward(self, x):
            identity = x
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity
            return self.relu(out)

    class PureResNet50(nn.Module):
        def __init__(self, num_classes=3):
            super().__init__()
            self.inplanes = 64
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(64, 3)
            self.layer2 = self._make_layer(128, 4, stride=2)
            self.layer3 = self._make_layer(256, 6, stride=2)
            self.layer4 = self._make_layer(512, 3, stride=2)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(512 * Bottleneck.expansion, num_classes)
            )

        def _make_layer(self, planes, blocks, stride=1):
            downsample = None
            if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * Bottleneck.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * Bottleneck.expansion)
                )
            layers = [Bottleneck(self.inplanes, planes, stride, downsample)]
            self.inplanes = planes * Bottleneck.expansion
            for _ in range(1, blocks):
                layers.append(Bottleneck(self.inplanes, planes))
            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            return self.fc(x)

    class RiceDiscolorationResNet50(nn.Module):
        def __init__(self, num_classes=3, pretrained=False):
            super().__init__()
            self.base = PureResNet50(num_classes=num_classes)

        def forward(self, x):
            return self.base(x)
else:
    class RiceDiscolorationResNet50:
        pass
