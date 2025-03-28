# -*- coding: utf-8 -*-
"""
Created on Sun Mar 07 19:48:35 2024

@author: MaxGr
"""

import os
import cv2
import time
import numpy as np
from collections import deque
from datetime import datetime
from ultralytics import YOLO
import torch


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
TORCH_CUDA_ARCH_LIST = "8.6"
current_datetime = datetime.now()
date_time_string = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

print('torch.version: ', torch.__version__)
print('torch.version.cuda: ', torch.version.cuda)
print('torch.cuda.is_available: ', torch.cuda.is_available())
print('torch.cuda.device_count: ', torch.cuda.device_count())
# print('torch.cuda.current_device: ', torch.cuda.current_device())
# device_default = torch.cuda.current_device()
# torch.cuda.device(device_default)
# print('torch.cuda.get_device_name: ', torch.cuda.get_device_name(device_default))
# device = torch.device("cuda")
device = torch.device("cpu")

'''
import models
'''
URBAN_WALKING_HAZARDS = [
    'person', 'cyclist', 'car', 'bus', 'motorcycle', 'scooter', 'fountain', 'bench',
    'traffic light', 'stop sign', 'curb', 'ramp', 'stair', 'escalator', 'charging station',
    'elevator', 'trash can', 'pole', 'tree', 'fire hydrant', 'lamp post', 'ATM', 'kiosk',
    'construction barrier', 'construction sign', 'scaffolding', 'hole', 'crack', 'speed bump',
    'puddle', 'manhole', 'drain', 'grate', 'loose gravel', 'ice patch', 'snow pile', 'leaf pile',
    'standing water', 'mud', 'sand', 'street sign', 'directional sign', 'information sign',
    'parking meter', 'mailbox', 'bicycle rack', 'outdoor seating', 'planter box', 'bollard',
    'guardrail', 'traffic cone', 'traffic barrel', 'pedestrian signal', 'crowd', 'animal', 'dog',
    'bird', 'cat', 'public restroom', 'fountain', 'statue', 'monument', 'picnic table',
    'outdoor advertisement', 'vendor cart', 'food truck', 'emergency exit', 'no smoking sign',
    'wet floor sign', 'closed sign', 'open sign', 'entrance sign', 'exit sign', 'stairs sign',
    'escalator sign', 'elevator sign', 'restroom sign', 'braille sign', 'audio signal device',
    'tactile paving', 'detectable warning surface', 'guide rail', 'handrail', 'turnstile',
    'gate', 'security checkpoint', 'water dispenser', 'vending machine',
    'public telephone', 'emergency phone',
    'first aid station', 'defibrillator',
    # Additional road hazards
    # 'uneven pavement', 
    'recently paved asphalt', 'oil spill', 'road debris', 'overhanging branches',
    'low-hanging signage', 'temporary road signs', 'roadworks', 'excavation sites', 'utility works',
    'fallen objects', 'spilled cargo', 'flood', 'ice', 'snowdrift', 'landslide debris',
    'erosion damage', 'parked vehicles', 'moving equipment',
    'street performers', 'demonstrations', 'large gatherings', 'parade', 'marathon', 'street fair',
    # 'crowded sidewalk', 'narrow sidewalk', 'blocked sidewalk', 
    'temporary scaffolding',
    'electrical hazards', 'wire tangle', 'unsecured manhole covers', 'improperly installed street elements',
    'visual distractions', 'audio distractions', 'smell hazards', 'toxic spill', 'biohazard materials',
    'wildlife crossings', 'stray animals', 'pets without leashes', 'flying debris', 'air pollution',
    'smoke plumes', 'dust storms', 'sandstorms', 'flash floods', 'earthquake damage', 'volcanic ash'
]

MASK = ['people', 'human face', 'car license plate', 'license plate', 'plate']

CLASSES = URBAN_WALKING_HAZARDS
weight_file = 'yolov8x-worldv2.pt'
model = YOLO(weight_file)
model.set_classes(CLASSES)

'''
Video setting
'''
text_color = [(0, 255, 0), (0, 0, 255)]
mark_danger = False

fps = 5
display_start_frame = 0
display_until_frame = 10000

# Open the video file
video_path = './Video/'
video_name = 'JP_1.MOV'

# Check if the video file opened successfully
video_capture = cv2.VideoCapture(video_path + video_name)
if not video_capture.isOpened():
    print("Error opening video file.")
    exit()  # Stop execution if there's an error

for i in range(1):
    ret, frame = video_capture.read()

# Get frame dimensions
img_height, img_width = frame.shape[:2]

# Calculate 'H' segmentation lines
left_line_x = img_width // 4
right_line_x = img_width * 3 // 4
top_line_y = img_height // 2
bottom_line_y = img_height // 2

output_video_filename = f"./output/output_video_{date_time_string}.mp4"

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter(output_video_filename, fourcc, 10.0, (img_height, img_width))

'''
detection
'''
skipped_frame = 6
motion_factor = 10  #60 // skipped_frame

object_list = []
object_alert = []

frame_id = 0
start_time = time.time()
while True:
    frame_id += 1

    print('Current frame: ', frame_id)
    ret, frame = video_capture.read()
    if not ret:  # End of the video
        break

    if frame_id < display_start_frame:
        continue
    if frame_id > display_until_frame:
        break

    cv2.putText(frame, f'{frame_id - i}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color[mark_danger], 2,
                cv2.LINE_AA)
    if frame_id % 5 in [1, 2, 3, 4]:
        continue

    '''
    yolo inference
    '''
    results = model.predict(frame)
    annotated_frame = results[0].plot()
    '''
    Display the frame
    '''
    # Draw lines for the 'H'
    cv2.line(annotated_frame, (left_line_x, 0), (left_line_x, img_height), (0, 255, 0), 10)  # Left vertical line
    cv2.line(annotated_frame, (right_line_x, 0), (right_line_x, img_height), (0, 255, 0), 10)  # Right vertical line
    cv2.line(annotated_frame, (left_line_x, top_line_y), (right_line_x, bottom_line_y), (0, 255, 0),
             10)  # Horizontal line

    cv2.imshow('Video Frame', annotated_frame)
    output_video.write(annotated_frame)

    time.sleep(1 / fps)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite('last.jpg', annotated_frame)
        break

video_capture.release()
output_video.release()

end_time = time.time()
total_time = end_time - start_time
FPS = (frame_id - display_start_frame) / total_time
print('FPS: ', FPS)

print('Unique Objects: ', len(object_list))
print('Danger Labels: ', len(object_alert))
