import torch

import cv2
import numpy as np

from models.model_feat_SEConvBlock_woPose import FeatureExtractor
from models.model_hm_3d_allLmksCat_pose_guided_att_SEConvBlock_dim9 import FAN
from models.model_coord_feats_in_4hg_1Res4HG_1HG_HPM2222_HMResAtt_input_hm3dFeat_hpm2222_inputPose_dim9_HMAtt_SE_ConvBlock import PCNet

from models.utils import colorization, DrawPose, RPY2Rot_torch

# basic setting
num_landmarks = 68

# model load
model_feat = FeatureExtractor()
model_hm = FAN(num_modules=4,
               end_relu=False,
               gray_scale=False,
               num_landmarks=num_landmarks)
model_coord = PCNet(num_landmarks)

model_ckpt = torch.load('./ckpt/pfal_heatmap_gt.pth', map_location='cpu')
model_feat.load_state_dict(model_ckpt['model_feat_state_dict'])
model_hm.load_state_dict(model_ckpt['model_hm_state_dict'])

model_feat.cuda()
model_hm.cuda()

model_feat.eval()
model_hm.eval()

with torch.no_grad():
    img = cv2.imread('./imgs/test_roll.png')
    img_src = img.copy()
    img = torch.tensor(img).transpose(0, -1).unsqueeze(0).cuda()

    for idx_angle in range(-45, 45):
        angle_add = torch.zeros((1, 3), dtype=torch.float32)
        angle_add[0][2] = idx_angle * 4
        angle_add = angle_add.cuda()

        _, _, _, x = model_feat(img)
        pose_modified = angle_add
        pose_modified = RPY2Rot_torch(pose_modified * torch.pi / 180.0)
        hm_out, feats = model_hm(x, pose_modified.view(-1, 9))

        pred_hm = cv2.resize(
            np.clip(hm_out[-1].transpose(2, 3).detach().cpu().numpy()[-1].max(0) * 255.0, 0.0, 255.0).astype(
                'uint8'), (256, 256))

        img_pose = np.ones((128, 128, 3), dtype='uint8') * 255
        img_pose = DrawPose(img_pose, rot=pose_modified[0].detach().cpu().numpy())
        img_result = colorization(pred_hm)
        img_pose = cv2.resize(img_pose, (64, 64))

        img_result = cv2.addWeighted(img_src, 0.5, img_result, 0.5, 0.0)
        img_result[192:, :64] = img_pose

        cv2.imwrite('./result/%06d.jpg' % (idx_angle + 45), img_result)

