# OpenMMLab Environment Setup 
 
This guide walks you through the steps to set up the OpenMMLab environment with the necessary dependencies. 
 
## Requirements 
 
- Conda 
- Python 3.8 
 
## Steps to Set Up 
 
### 1. Create and Activate the Conda Environment 
First, create a new conda environment with Python 3.8: 
 
```bash 
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

### 2. Install PyTorch and Dependencies
Next, install PyTorch version 1.9.0 along with torchvision and the necessary CUDA toolkit:

```bash
conda install pytorch=1.9.0 torchvision=0.10.0 cudatoolkit=11.1 -c pytorch
```

### 3. Install OpenMMLab and Other Dependencies
Now install the OpenMMLab tools and libraries using mim (OpenMMLab installation manager):

```bash
pip install openmim==0.3.9
mim install mmengine==0.10.3
mim install mmcv==2.0.1
mim install mmdet==3.2.0
```

### 4. Install Additional Libraries
Install other required libraries:

```bash
pip install timm
pip install segmentation-models-pytorch
mim install mmyolo==0.6.0
mim install mmpose==1.3.1
pip install segment-anything
```

### 5. Verify Installation
To verify your installation, you can run the following Python commands:

```python
import torch
import mmcv
import mmdet
import mmyolo

print(f"PyTorch version: {torch.__version__}")
print(f"MMCV version: {mmcv.__version__}")
print(f"MMDetection version: {mmdet.__version__}")
print(f"MMYOLO version: {mmyolo.__version__}")
```

### 6. Download Checkpoints

You need to download the checkpoints from the official MMYOLO repository. You can download the YOLOv8 checkpoint from the following link: https://github.com/open-mmlab/mmyolo/tree/main/configs/yolov8

Additionally, you need to download the DPT checkpoint from the MiDaS repository:
https://github.com/isl-org/MiDaS/releases/download/v3/dpt_hybrid_384.pt

### 7. Download Demo Video
To test the installation, you can download the demo video from the following link:
https://scontent.xx.fbcdn.net/m1/v/t6/An84biPUErSZePAyqjTvsNKRwvNAvU8aXaPJ9ag7S-wcgv1zrMrCQM0jpDRPQp3GwLgbc2Bkyh3TvvSmwYzSmtNratn9LW9JMLiJfFNJXi-id0ZqXlLBo3sHjlmFOE6jTbpY6RwVSiEYkl1RC0GJmMcF1fwAkTFydwk1Ts7MklSokQ.mp4/AriaEverydayActivities_1.0.0_loc3_script4_seq7_rec1_preview_rgb.mp4?ccb=10-5&oh=00_AYBi2SfLr6YFf_uN4_s6Kz4107uGrvgOrpTqmiM5sq9pvA&oe=67D66A75&_nc_sid=792287

### 8. Run Demo
To test the installation, you can run a demo using a sample video:

```bash
python demo_video.py
```

### 9. Notes

- The versions specified are known to work together
- If you need to use different versions, make sure to check the compatibility matrix in the OpenMMLab documentation
- For GPU support, ensure you have the correct NVIDIA drivers installed

