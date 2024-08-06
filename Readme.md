# Pose-fused Face Alignment
![graphical_abstract](https://github.com/user-attachments/assets/39639ffb-4648-446d-9d7d-c9d233cc30c1)


## Example
### Pose-fused 3D Heatmap Regression

https://github.com/user-attachments/assets/f444db6e-6813-41ed-9f87-739a92989abb
#
Run Heatmap Regression on Test Images

1. Download Models [Google Drive](https://drive.google.com/file/d/1HS7TMExYlJHc4ojrwrQ48BYE60yxwCmb/view?usp=sharing) and put it in ```./ckpt``` directory.
2. Run test_inputpose_heatmap.py
#

Source:
#
Left: https://www.pexels.com/photo/woman-lying-on-a-bathtub-7278795/

Center: https://www.pexels.com/photo/woman-wearing-black-top-and-blue-denim-bottoms-2010877/

Right: https://www.pexels.com/photo/portrait-shot-of-a-woman-2817080/

#
### Pose-fused 3D Dense Face Alignment
Red: Keypoint Landmarks, Blue(Brighter: Near): Geometric Landmarks
#
https://github.com/user-attachments/assets/97fe3578-bcc0-4f1b-b7a0-d2e26eddef6e

Source1: https://www.pexels.com/video/a-woman-in-the-corner-area-dancing-in-poses-for-a-photo-shoot-3403330/
#
https://github.com/user-attachments/assets/7b30928a-eca6-4cbc-ac12-b4ab9b64f2fb

Source2: https://www.pexels.com/video/woman-massaging-her-face-with-jade-roller-6933178/
#

## Citation
If you find this code useful in your research, please consider citing:
```
@misc{so20233dfacealignmentfusion,
      title={3D Face Alignment Through Fusion of Head Pose Information and Features}, 
      author={Jaehyun So and Youngjoon Han},
      year={2023},
      eprint={2308.13327},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2308.13327}, 
}
```

## Future Plans
- [ ] Release demo code and pretrained weights on 300W-LP dataset
- [ ] Release evaluation code on AFLW2000 dataset
- [ ] Release training code on 300W-LP dataset
