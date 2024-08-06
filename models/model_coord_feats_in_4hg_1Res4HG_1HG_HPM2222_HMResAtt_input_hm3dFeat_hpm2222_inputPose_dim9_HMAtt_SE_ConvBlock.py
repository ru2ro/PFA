import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import SE_ConvBlock3D_dim9, SE_ConvBlock_inputPose_dim9

class HourGlass(nn.Module):
    def __init__(self, num_modules, depth, num_features):
        super(HourGlass, self).__init__()
        self.num_modules = num_modules
        self.depth = depth
        self.features = num_features
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
            low2 = self._forward(level - 1, low1, pose)
        else:
            low2 = low1
            low2 = self._modules['b2_plus_' + str(level)](low2, pose)

        low3 = low2
        low3 = self._modules['b3_' + str(level)](low3, pose)

        up2 = F.upsample(low3, scale_factor=2, mode='nearest')
        return up1 + up2

    def forward(self, x, pose):
        return self._forward(self.depth, x, pose)


class CoordNet(nn.Module):
    def __init__(self, num_landmarks, num_layers_f=4):
        super(CoordNet, self).__init__()

        self.num_layers_f = num_layers_f
        self.num_landmarks = num_landmarks
        num_feat_f = 256
        num_feat_h_in = [64, 128, 256, 512]
        num_feat_h_out = [128, 256, 512, 256]

        self.conv_f_1_1 = SE_ConvBlock_inputPose_dim9(num_feat_f, num_feat_f)
        self.conv_f_1_2 = SE_ConvBlock_inputPose_dim9(num_feat_f, num_feat_f)
        self.conv_f_2_1 = SE_ConvBlock_inputPose_dim9(num_feat_f, num_feat_f)
        self.conv_f_2_2 = SE_ConvBlock_inputPose_dim9(num_feat_f, num_feat_f)
        self.conv_f_3_1 = SE_ConvBlock_inputPose_dim9(num_feat_f, num_feat_f)
        self.conv_f_3_2 = SE_ConvBlock_inputPose_dim9(num_feat_f, num_feat_f)
        self.conv_f_4_1 = SE_ConvBlock_inputPose_dim9(num_feat_f, num_feat_f)
        self.conv_f_4_2 = SE_ConvBlock_inputPose_dim9(num_feat_f, num_feat_f)
        self.bn_f = nn.Sequential(nn.BatchNorm2d(num_feat_f), nn.ReLU(inplace=True))

        self.conv_h = nn.Conv3d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.conv_h_1_1 = SE_ConvBlock3D_dim9(num_feat_h_in[0], num_feat_h_out[0])
        self.conv_h_1_2 = SE_ConvBlock3D_dim9(num_feat_h_out[0], num_feat_h_out[0])
        self.conv_h_2_1 = SE_ConvBlock3D_dim9(num_feat_h_in[1], num_feat_h_out[1])
        self.conv_h_2_2 = SE_ConvBlock3D_dim9(num_feat_h_out[1], num_feat_h_out[1])
        self.conv_h_3_1 = SE_ConvBlock3D_dim9(num_feat_h_in[2], num_feat_h_out[2])
        self.conv_h_3_2 = SE_ConvBlock3D_dim9(num_feat_h_out[2], num_feat_h_out[2])
        self.conv_h_4_1 = SE_ConvBlock3D_dim9(num_feat_h_in[3], num_feat_h_out[3])
        self.conv_h_4_2 = SE_ConvBlock3D_dim9(num_feat_h_out[3], num_feat_h_out[3])
        self.bn_h = nn.Sequential(nn.BatchNorm3d(num_feat_h_out[-1]), nn.ReLU(inplace=True))

        self.fc_coord_1 = nn.Linear(num_feat_f + num_feat_h_out[-1], self.num_landmarks * 3)

    def forward(self, x, h, pose):

        x = F.avg_pool2d(x, 2)
        x = self.conv_f_1_1(x, pose)
        x = self.conv_f_1_2(x, pose)
        x = F.avg_pool2d(x, 2)
        x = self.conv_f_2_1(x, pose)
        x = self.conv_f_2_2(x, pose)
        x = F.avg_pool2d(x, 2)
        x = self.conv_f_3_1(x, pose)
        x = self.conv_f_3_2(x, pose)
        x = F.avg_pool2d(x, 2)
        x = self.conv_f_4_1(x, pose)
        x = self.conv_f_4_2(x, pose)
        x = self.bn_f(x)

        h = self.conv_h(h.unsqueeze(1))
        h = F.avg_pool3d(h, 2)
        h = self.conv_h_1_1(h, pose)
        h = self.conv_h_1_2(h, pose)
        h = F.avg_pool3d(h, 2)
        h = self.conv_h_2_1(h, pose)
        h = self.conv_h_2_2(h, pose)
        h = F.avg_pool3d(h, 2)
        h = self.conv_h_3_1(h, pose)
        h = self.conv_h_3_2(h, pose)
        h = F.avg_pool3d(h, 2)
        h = self.conv_h_4_1(h, pose)
        h = self.conv_h_4_2(h, pose)
        h = self.bn_h(h)

        x = F.avg_pool2d(x, 4)
        h = F.avg_pool3d(h, 4)
        x = torch.flatten(x, 1)
        h = torch.flatten(h, 1)
        x = torch.cat([x, h], dim=1)
        coord = self.fc_coord_1(x)
        coord = coord.view(-1, self.num_landmarks, 3)

        return coord, x


class PCNet(nn.Module):
    def __init__(self, num_landmarks, repeat=3, num_layer=4):
        super(PCNet, self).__init__()
        self.num_landmarks = num_landmarks
        self.repeat = repeat
        num_feat_h = 256

        self.add_module('Conv_pre', nn.Sequential(nn.BatchNorm2d(4 * num_feat_h),
                                                  nn.ReLU(inplace=True),
                                                  nn.Conv2d(4 * num_feat_h, num_feat_h, kernel_size=1, stride=1, padding=0, bias=False)))
        self.add_module('HMAtt', SE_ConvBlock_inputPose_dim9(64 + num_feat_h, num_feat_h))

        self.HG1 = HourGlass(num_modules=1, depth=4, num_features=256)

        self.conv_last = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(True))

        self.CoordReg = CoordNet(num_landmarks, num_layer)

    def forward(self, x_f, heatmap, pose):
        x_f.reverse()
        for i in range(1):
            x_f[i] = self._modules['Conv_pre'](x_f[i])
            hm_cur = F.avg_pool2d(heatmap, heatmap.shape[-1] // x_f[i].shape[-1])
            att = self._modules['HMAtt'](torch.cat([x_f[i], hm_cur], dim=1), pose)
            x_f[i] = x_f[i] * att

        feats = self.HG1(x_f[0], pose)
        feats = self.conv_last(feats)

        coord = self.CoordReg(feats, heatmap, pose)

        return coord
