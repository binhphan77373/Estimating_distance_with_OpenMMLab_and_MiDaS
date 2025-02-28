import os
import cv2
import torch
import torchvision.transforms as transforms
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time

import torchvision
import timm
import segmentation_models_pytorch as smp
from torchvision.transforms import ToTensor
from PIL import Image, ImageDraw
from mmdet.apis import async_inference_detector, inference_detector
from mmdet.apis.inference import init_detector

from utils import *
from segment_anything import sam_model_registry, SamPredictor

from dpt.models import DPTDepthModel
from dpt.midas_net import MidasNet_large
from dpt.transforms import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose

# Initialize the person detection model
person_mmdet_config_file = "/home/eyecode-binh/Estimating_distance_with_OpenMMLab_and_MiDaS/checkpoint_distance/yolov8_x_mask-refine_syncbn_fast_8xb16-500e_coco.py"
person_mmdet_checkpoint = '/home/eyecode-binh/Estimating_distance_with_OpenMMLab_and_MiDaS/checkpoint_distance/yolov8_x_mask-refine_syncbn_fast_8xb16-500e_coco_20230217_120411-079ca8d1.pth'
person_mmdet_model = init_detector(person_mmdet_config_file, person_mmdet_checkpoint, device='cuda:1')

# Path to the video file
path_video = '/home/eyecode-binh/Estimating_distance_with_OpenMMLab_and_MiDaS/video_demo/AriaEverydayActivities.mp4'
output_video_path = '/home/eyecode-binh/Estimating_distance_with_OpenMMLab_and_MiDaS/video_demo/output_video_aria.mp4'

# Initialize video capture
cap = cv2.VideoCapture(path_video)

# Get video properties for VideoWriter
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Initialize VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Initialize the depth estimation model
model_path = "/home/eyecode-binh/Estimating_distance_with_OpenMMLab_and_MiDaS/checkpoint_dpt/dpt_hybrid_384.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net_w = 1400
net_h = 1400

model = DPTDepthModel(
    path=model_path,
    scale=0.000305,
    shift=0.1378,
    invert=True,
    backbone="vitb_rn50_384",
    non_negative=True,
    enable_attention_hooks=False,
)

normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

model.eval()
model = model.to(memory_format=torch.channels_last)
model = model.half()
model.to(device)

# Function to filter classes
def filter_class(raw_worker_bboxes, raw_worker_labels, raw_worker_scores, class_names, prefix='chair'):
    worker_bboxes = []
    worker_labels = []
    worker_scores = []
    for i, worker_bbox in enumerate(raw_worker_bboxes):
        if raw_worker_labels[i] in class_names:
            worker_bboxes.append(worker_bbox)
            worker_labels.append(prefix)
            worker_scores.append(raw_worker_scores[i])
    return worker_bboxes, worker_labels, worker_scores

# Function to estimate depth
def depth_estimation_dp(img):
    start_time = time.time()
    img_input = transform({"image": img})["image"]
    with torch.no_grad():
        sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
        sample = sample.to(memory_format=torch.channels_last)
        sample = sample.half()
        prediction = model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )
    end_time = time.time()
    execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
    prediction *= 100.0
    return prediction, execution_time

# Function to extract depth from bounding boxes
def extract_depth_from_boxes(boxes, depth_map):
    object_depths = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        object_depth_map = depth_map[y1:y2, x1:x2]
        median_depth = np.mean(object_depth_map)
        object_depths.append(median_depth)
    return object_depths

# Function to extract bounding boxes and depth
def extract_bounding_boxes_and_depth(detected_boxes, detected_labels, depths):
    objects = []
    for i, box in enumerate(detected_boxes):
        x1, y1, x2, y2 = map(int, box)
        obj = {
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'class': detected_labels[i],
            'depth': depths[i]
        }
        objects.append(obj)
    return objects

# Function to compute chair to camera distance
def compute_chair_to_camera_distance(objects):
    chair_distances = {}
    for i, obj in enumerate(objects):
        if obj['class'] == 'chair':
            chair_distances[i] = {'distance_to_camera': obj['depth']}
    return chair_distances

# Variables to track metrics
execution_times = []
total_frames = 0

# Process each frame in the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    total_frames += 1
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    dpt_frame = frame / 255.0

    # Detect objects in the frame
    worker_threshold = 0.5
    worker_results = inference_detector(person_mmdet_model, frame)
    raw_worker_bboxes, raw_worker_labels, raw_worker_scores = mmdet3x_convert_to_bboxes_mmdet(worker_results, worker_threshold)
    worker_bboxes, worker_labels, worker_scores = filter_class(raw_worker_bboxes, raw_worker_labels, raw_worker_scores, 'class_57', prefix='chair')

    # Estimate depth
    depth_map, execution_time = depth_estimation_dp(dpt_frame)
    execution_times.append(execution_time)

    # Extract depth from bounding boxes
    object_depths = extract_depth_from_boxes(worker_bboxes, depth_map)
    object_infor = extract_bounding_boxes_and_depth(worker_bboxes, worker_labels, object_depths)

    # Compute chair to camera distance
    object_distances = compute_chair_to_camera_distance(object_infor)
    center_points = [((box['x1'] + box['x2']) // 2, (box['y1'] + box['y2']) // 2)
                     for box in object_infor]

    # Visualize results
    vis_results = frame.copy()
    violated_worker = 0
    for i, distance_info in object_distances.items():
        pt1 = center_points[i]
        distance = distance_info['distance_to_camera']

        # Display distance on the image
        cv2.putText(vis_results, f"{distance:.2f}cm", pt1, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # Draw line from camera to object
        cv2.line(vis_results, (vis_results.shape[1]//2, vis_results.shape[0]), pt1, (0, 0, 255), 2)
        if distance < 100:
            violated_worker += 1

    # Display current metrics on frame
    avg_exec_time = np.mean(execution_times) if execution_times else 0
    cv2.putText(vis_results, f"Execution time: {avg_exec_time:.2f}ms", (30, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(vis_results, f"Frames processed: {total_frames}", (30, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Convert back to BGR for saving
    vis_results = cv2.cvtColor(vis_results, cv2.COLOR_RGB2BGR)
    out.write(vis_results)

# Calculate final metrics
avg_exec_time = np.mean(execution_times) if execution_times else 0

# Print final metrics
print(f"Average execution time for Disparity Estimation with DPT_Hybrid: {avg_exec_time:.2f} ms")
print(f"Total frames processed: {total_frames}")

cap.release()
out.release()
print(f"Output video saved to {output_video_path}")  