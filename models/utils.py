import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import cv2


def DrawPose(img, pts=None, rot=None, num_landmarks=None):
    len_line = 50
    pose = np.zeros((3, 3), dtype='float64')
    pose[0, 0] = len_line
    pose[1, 1] = len_line
    pose[2, 2] = len_line

    rstPoints = []
    if num_landmarks == 98:
        rstPoints.append(pts[51])
    elif num_landmarks == 19:
        pt_draw = pts[7] + pts[10]
        pt_draw[0] = int((pt_draw[0] + pts[13, 0]) / 3)
        pt_draw[1] = int(pt_draw[1] / 2)
        rstPoints.append(np.array(pt_draw))
    elif num_landmarks == 5:
        pt_draw = pts[0] + pts[1]
        pt_draw[0] = int((pt_draw[0] + pts[2, 0]) / 3)
        pt_draw[1] = int(pt_draw[1] / 2)
        rstPoints.append(np.array(pt_draw))
    elif num_landmarks == 68:
        rstPoints.append(np.array(pts[27]))
    else:
        rstPoints.append(np.array((img.shape[1] // 2, img.shape[0] // 2)))

    rstPoints[0] = np.array([rstPoints[0][0], rstPoints[0][1], 0.0])
    for idx in range(3):
        pose_cur = rot.dot(pose[idx])
        rstPoints.append((rstPoints[0] + pose_cur))
    rstPoints = np.array(rstPoints, dtype='int32')
    idx_order = np.argsort(rstPoints[1:].transpose(1, 0)[2])
    for i in idx_order[::-1].tolist():
        color = np.zeros(3, dtype='int')
        color[i] = 255
        cv2.line(img, (rstPoints[0][0], rstPoints[0][1]), (rstPoints[i+1][0], rstPoints[i+1][1]),
                 (int(color[0]), int(color[1]), int(color[2])), 4)

    return img

def colorization(x):
    size_y = x.shape[0]
    size_x = x.shape[1]
    y = 2 * np.pi * (x - x.min()) / (x.max() - x.min())
    val_min = 52 * np.pi / 180.0
    y[y < val_min] = val_min

    y_c = np.zeros((size_y, size_x, 3), dtype=np.float32)
    y_c[:, :, 0] = +1.0 * np.sin(y)
    y_c[:, :, 1] = -1.0 * np.cos(y)
    y_c[:, :, 2] = -1.0 * np.sin(y)
    y_c = (127.5 * (y_c + 1)).astype('uint8')
    return y_c

def RPY2Rot_torch(rpy):
    wuv_cos = torch.cos(rpy)
    wuv_sin = torch.sin(rpy)

    cu = wuv_cos[:, 0]
    cv = wuv_cos[:, 1]
    cw = wuv_cos[:, 2]
    su = wuv_sin[:, 0]
    sv = wuv_sin[:, 1]
    sw = wuv_sin[:, 2]

    e11 = cw * cv
    e12 = cw * sv * su - sw * cu
    e13 = cw * cu * sv + sw * su
    e21 = sw * cv
    e22 = sw * sv * su + cw * cu
    e23 = sw * sv * cu - cw * su
    e31 = -sv
    e32 = cv * su
    e33 = cv * cu

    r1 = torch.cat([e11.unsqueeze(1), e12.unsqueeze(1), e13.unsqueeze(1)], dim=1)
    r2 = torch.cat([e21.unsqueeze(1), e22.unsqueeze(1), e23.unsqueeze(1)], dim=1)
    r3 = torch.cat([e31.unsqueeze(1), e32.unsqueeze(1), e33.unsqueeze(1)], dim=1)

    return torch.cat([r1.unsqueeze(1), r2.unsqueeze(1), r3.unsqueeze(1)], dim=1)


def conv3x3(in_planes, out_planes, strd=1, padding=1, groups=1,
            bias=False, dilation=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=strd, padding=padding, groups=groups,
                     bias=bias, dilation=dilation)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SE_Attention_Pose_Input_dim9(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SE_Attention_Pose_Input_dim9, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_sq = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True)
        )

        self.fc_ex = nn.Sequential(
            nn.Linear(channel // reduction + 9, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, pose):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc_sq(y)
        y = torch.cat([y, pose], dim=1)
        y = self.fc_ex(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SE_ConvBlock_woPose(nn.Module):
    def __init__(self, in_planes, out_planes, groups=1, padding=1, dilation=1):
        super(SE_ConvBlock_woPose, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, int(out_planes / 2), padding=padding, dilation=dilation, groups=groups)
        self.bn2 = nn.BatchNorm2d(int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4),
                             padding=padding, dilation=dilation, groups=groups)
        self.bn3 = nn.BatchNorm2d(int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4),
                             padding=padding, dilation=dilation, groups=groups)

        self.SE = SELayer(out_planes)

        self.groups = groups
        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.BatchNorm2d(in_planes),
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes,
                          kernel_size=1, stride=1, bias=False),
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)

        out3 = torch.cat((out1, out2, out3), 1)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out3 = self.SE(out3)
        out3 += residual

        return out3


class SE_Attention3D_Pose_Input_dim9(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SE_Attention3D_Pose_Input_dim9, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc_sq = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True)
        )

        self.fc_ex = nn.Sequential(
            nn.Linear(channel // reduction + 9, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, pose):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc_sq(y)
        y = torch.cat([y, pose], dim=1)
        y = self.fc_ex(y).view(b, c, 1, 1, 1)

        return x * y.expand_as(x)


class SE_ConvBlock_inputPose_dim9(nn.Module):
    def __init__(self, in_planes, out_planes, groups=1, padding=1, dilation=1):
        super(SE_ConvBlock_inputPose_dim9, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, int(out_planes / 2), padding=padding, dilation=dilation, groups=groups)
        self.bn2 = nn.BatchNorm2d(int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4),
                             padding=padding, dilation=dilation, groups=groups)
        self.bn3 = nn.BatchNorm2d(int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4),
                             padding=padding, dilation=dilation, groups=groups)

        self.SE_pose = SE_Attention_Pose_Input_dim9(out_planes)

        self.groups = groups
        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.BatchNorm2d(in_planes),
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes,
                          kernel_size=1, stride=1, bias=False),
            )
        else:
            self.downsample = None

    def forward(self, x, pose):
        residual = x

        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)

        out3 = torch.cat((out1, out2, out3), 1)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out3 = self.SE_pose(out3, pose)
        out3 += residual

        return out3

class SE_ConvBlock3D_dim9(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, groups=1):
        super(SE_ConvBlock3D_dim9, self).__init__()
        self.bn1 = nn.BatchNorm3d(in_planes)
        self.conv1 = nn.Conv3d(in_planes, int(out_planes / 2), kernel_size=kernel_size, stride=stride, padding=1,
                               groups=groups, bias=False, dilation=1)
        self.bn2 = nn.BatchNorm3d(int(out_planes / 2))
        self.conv2 = nn.Conv3d(int(out_planes / 2), int(out_planes / 4), kernel_size=kernel_size, stride=stride,
                               padding=padding, groups=groups, bias=False, dilation=1)
        self.bn3 = nn.BatchNorm3d(int(out_planes / 4))
        self.conv3 = nn.Conv3d(int(out_planes / 4), int(out_planes / 4), kernel_size=kernel_size, stride=stride,
                               padding=padding, groups=groups, bias=False, dilation=1)

        self.SE_pose = SE_Attention3D_Pose_Input_dim9(out_planes)

        self.groups = groups
        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.BatchNorm3d(in_planes),
                nn.ReLU(True),
                nn.Conv3d(in_planes, out_planes,
                          kernel_size=1, stride=1, bias=False),
            )
        else:
            self.downsample = None

    def forward(self, x, pose):
        residual = x

        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)

        out3 = torch.cat((out1, out2, out3), 1)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out3 = self.SE_pose(out3, pose)
        out3 += residual

        return out3
