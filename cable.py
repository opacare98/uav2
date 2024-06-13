#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 10:09:39 2024

@author: chushiyun
"""

from ultralytics import YOLO
import cv2
import time
import torch.distributed as dist

"""
dist.init_process_group('nccl', #init_method='file:///mnt/data/uav-dataset', 
                        rank=0, world_size=1)
"""

# Load the model.
model = YOLO('yolov8n.pt')


# Training.
results = model.train(
   #data='uav.yaml',
   data='cable.yaml',
   imgsz=2000,
   epochs=2,
   batch=5,
   name='yolov8n_v8_60e'
)


