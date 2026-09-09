import torch
from torch import nn


class GenreCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        blocks = []
        incoming = 1
        for outgoing in [16, 32, 64, 128]:
            blocks.extend(
                [
                    nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False),
                    nn.BatchNorm2d(outgoing),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Dropout2d(0.1),
                ]
            )
            incoming = outgoing
        self.features = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def augment(x):
    """Frequency/time masking, on training batches only."""
    x = x.clone()
    for sample in x:
        for axis, max_width in [(1, 8), (2, 16)]:
            width = int(torch.randint(0, max_width + 1, ()).item())
            if width:
                start = int(torch.randint(0, sample.shape[axis] - width + 1, ()).item())
                if axis == 1:
                    sample[:, start : start + width, :] = -1
                else:
                    sample[:, :, start : start + width] = -1
    return x
