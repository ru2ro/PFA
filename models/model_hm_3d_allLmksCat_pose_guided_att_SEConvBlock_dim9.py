import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import SE_ConvBlock_inputPose_dim9
from models.coord_conv import CoordConvTh

class HourGlass(nn.Module):
    def __init__(self, num_modules, depth, num_features, first_one=False):
        super(HourGlass, self).__init__()
        self.num_modules = num_modules
        self.depth = depth
        self.features = num_features
        self.coordconv = CoordConvTh(x_dim=64, y_dim=64,
                                     with_r=True, with_boundary=False,
                                     in_channels=256, first_one=first_one,
                                     out_channels=256,
                                     kernel_size=1,
                                     stride=1, padding=0)
        self._generate_network(self.depth)

    def _generate_network(self, level):
        self.add_module('b1_' + str(level), SE_ConvBlock_inputPose_dim9(256, 256))

        self.add_module('b2_' + str(level), SE_ConvBlock_inputPose_dim9(256, 256))

        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), SE_ConvBlock_inputPose_dim9(256, 256))

        self.add_module('b3_' + str(level), SE_ConvBlock_inputPose_dim9(256, 256))

    def _forward(self, level, inp, pose):
        # Upper branch
        up1 = inp
        up1 = self._modules['b1_' + str(level)](up1, pose)

        # Lower branch
        low1 = F.avg_pool2d(inp, 2, stride=2)
        low1 = self._modules['b2_' + str(level)](low1, pose)

        if level > 1:
            low2, lows = self._forward(level - 1, low1, pose)
        else:
            low2 = low1
            low2 = self._modules['b2_plus_' + str(level)](low2, pose)
            lows = list()
            lows.append(low1)

        low3 = low2
        low3 = self._modules['b3_' + str(level)](low3, pose)

        up2 = F.upsample(low3, scale_factor=2, mode='nearest')

        lows.append(up1 + up2)
        # return up1 + up2, lows
        return lows[-1], lows

    def forward(self, x, heatmap, pose):
        x, last_channel = self.coordconv(x, heatmap)
        return self._forward(self.depth, x, pose), last_channel


class FAN(nn.Module):
    def __init__(self, num_modules=4, end_relu=False, gray_scale=False,
                 num_landmarks=68):
        super(FAN, self).__init__()
        self.num_modules = num_modules
        self.gray_scale = gray_scale
        self.end_relu = end_relu
        self.num_landmarks = num_landmarks

        # Stacking part
        for hg_module in range(self.num_modules):
            if hg_module == 0:
                first_one = True
            else:
                first_one = False
            self.add_module('m' + str(hg_module), HourGlass(1, 4, 256,
                                                            first_one))
            self.add_module('top_m_' + str(hg_module), SE_ConvBlock_inputPose_dim9(256, 256))
            self.add_module('conv_last' + str(hg_module),
                            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
            self.add_module('bn_end' + str(hg_module), nn.BatchNorm2d(256))
            self.add_module('l' + str(hg_module), nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0))

            if hg_module < self.num_modules - 1:
                self.add_module(
                    'bl' + str(hg_module), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
                self.add_module('al' + str(hg_module), nn.Conv2d(64,
                                                                 256, kernel_size=1, stride=1, padding=0))

    def forward(self, x, pose):
        previous = x

        outputs = []
        boundary_channels = []
        feats = []
        tmp_out = None

        for i in range(self.num_modules):
            # CoordConv, Hourglass
            hg, boundary_channel = self._modules['m' + str(i)](previous, tmp_out, pose)

            ll = hg[0]

            # Residual
            ll = self._modules['top_m_' + str(i)](ll, pose)

            # FC
            ll = F.relu(self._modules['bn_end' + str(i)]
                        (self._modules['conv_last' + str(i)](ll)), False)

            # Predict heatmaps, Score
            tmp_out = self._modules['l' + str(i)](ll)

            # end_relu: DEFAULT: False
            if self.end_relu:
                tmp_out = F.relu(tmp_out)  # HACK: Added relu
            outputs.append(tmp_out)
            boundary_channels.append(boundary_channel)

            if i < self.num_modules - 1:
                ll = self._modules['bl' + str(i)](ll)
                tmp_out_ = self._modules['al' + str(i)](tmp_out)
                previous = previous + ll + tmp_out_

            feats.append(hg[1])

        feat1, feat2, feat3, feat4 = feats
        feats = []
        feats.append(torch.cat([feat1[0], feat2[0], feat3[0], feat4[0]], dim=1))
        feats.append(torch.cat([feat1[1], feat2[1], feat3[1], feat4[1]], dim=1))
        feats.append(torch.cat([feat1[2], feat2[2], feat3[2], feat4[2]], dim=1))
        feats.append(torch.cat([feat1[3], feat2[3], feat3[3], feat4[3]], dim=1))
        feats.append(torch.cat([feat1[4], feat2[4], feat3[4], feat4[4]], dim=1))

        return outputs, feats