import torch.nn as nn
import torch.nn.functional as F
from models.utils import  SE_ConvBlock_woPose
from models.coord_conv import CoordConvTh


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        # Base part
        self.conv1 = CoordConvTh(x_dim=256, y_dim=256,
                                 with_r=True, with_boundary=False,
                                 in_channels=3, out_channels=64,
                                 kernel_size=7,
                                 stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = SE_ConvBlock_woPose(64, 128)
        self.layer2 = SE_ConvBlock_woPose(128, 128)
        self.layer3 = SE_ConvBlock_woPose(128, 256)

    def forward(self, x):
        # CoordConv
        x = x.float()
        x, _ = self.conv1(x)
        x = F.relu(self.bn1(x), False)
        x1 = x.clone()

        # Residual Unit + Down Sampling
        x = self.layer1(x)
        x = F.avg_pool2d(x, 2, stride=2)
        x2 = x.clone()

        # Residual Unit
        x = self.layer2(x)
        x3 = x.clone()

        x = self.layer3(x)
        x4 = x.clone()

        x1 = F.interpolate(x1, scale_factor=0.5)
        return x1, x2, x3, x4