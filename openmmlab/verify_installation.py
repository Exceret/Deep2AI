"""Verify installation."""

import os
from mmdet.apis import init_detector, inference_detector

os.chdir("openmmlab")
print(f"working directory: {os.getcwd()}")


config_file = "rtmdet_tiny_8xb32-300e_coco.py"
checkpoint_file = "rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth"
model = init_detector(
    config_file, checkpoint_file, device="cuda:0"
)  # or device='cuda:0'
infer_res = inference_detector(model, "demo.jpg")
