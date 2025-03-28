import os
import cv2
import time
import numpy as np
from collections import deque
from datetime import datetime
from ultralytics import YOLO
import torch

video_path = './Video/'
video_name = 'jp1.mp4'

video_capture = cv2.VideoCapture(video_path + video_name)
if not video_capture.isOpened():
    print('Cannot open video file')
    exit()
rotate_tag = video_capture.get(cv2.CAP_PROP_ORIENTATION_META)
print(rotate_tag)
frame_count = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
print(frame_count)

ret, frame = video_capture.read()
print(frame.shape[:2])

