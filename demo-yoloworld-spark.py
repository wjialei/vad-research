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
from ultralytics import RTDETR
import torch
from openai import OpenAI
from func_timeout import func_set_timeout
import threading


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
TORCH_CUDA_ARCH_LIST = "8.6"
current_datetime = datetime.now()
date_time_string = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

print('torch.version: ', torch.__version__)
print('torch.version.cuda: ', torch.version.cuda)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

system_sensitivity = 'normal'

prompt_background = "{You are an voice assistant for blind person, \
the input is the actual data collected by a phone camera, the phone is always facing front, \
please provide the key information for the blind user to help him navigate and avoid potential danger. \
Please note that the xloc and yloc represent the object location (proportional to the image), \
object height and width are also a proportion.}"
    
prompt_location = "{The location information (center_x, center_y, height, width) of objects is the proportion to the image, \
the detected objects are categorized into 4 type based on the image region.\
Left and Right: objects located on left 25% or right 25% of the image, these objects are usually moving and has large proportion.\
Front: objects that are still far away, can be used to discriminate the current situation.\
Ground: objects that are nearby, need to be cautioned.}"

prompt_motion = "{Analyze the movement (speed and direction) \
and location (xloc and yloc) of each object to determine its trajectory relative to the user.\
Use this information to assess whether an object is moving towards the user and if so, \
how quickly a potential collision might occur based on the object's speed and direction of movement.}"
    
prompt_sensitivity = '{System sensitivity: Incorporate the sensitivity setting in your response. \
For a low sensitivity setting, identify and report only imminent and direct threats to safety. \
For normal sensitivity, include potential hazards that could pose a risk if not avoided. \
For high sensitivity, report all detected objects that could potentially cause any level of inconvenience or danger.\
More focus on pedestrians and less focus on cars, as users are mostly walking on the sidewalk. \
Please more focus on the left,right,and ground area, as they are usually very close,\
but when you evaluate the emergency, consider the size and type of objects.\
Current sensitivity: ' + str(system_sensitivity) + '}'

instruction = prompt_background + prompt_location + prompt_sensitivity

prompt_format_full = 'Please organize your output into this format: \
{ "scene": …, quickly describe the current situation for blind user; \
  "key_objects" …, quickly and roughly locate the key objects for blind user; \
  "danger_checker": …, quickly diagnose if there is potential danger for a blind person; \
  "danger_label": …, output 1 if there is an emergency, output 0 if not; \
  "danger_index": [object_id, danger_index], estimate a score from 0 to 100 about each objects that may cause danger; \
  "voice_guide": …, the main output to instant alert the blind person for emergency.}'

prompt_format_turbo = 'Please organize your output into this format: \
{ "danger_score": output 1 for immediate threat, output 0 if not; \
  "reason": the main output to instant alert the blind person for emergency.}'
    
prompt_format_benchmark = 'Please organize your output into this format: \
{ "danger_score": output 1 for immediate threat, output 0 if not; \
  "reason": explain your annotation reason within 10 words.}'
    
# { "danger_score": predict a score from 0 to 1 to evaluate the emergency level, non-emergency shoule below 0.5; \

prompt_word_limiter = 'Limit your answer into 20 words'

load_start_time = time.time()
weight_file = 'yolov8x-worldv2.pt'
model = YOLO(weight_file)
# weight_file = 'rtdetr-l.pt'
# model = RTDETR(weight_file)
model.set_classes(CLASSES)
load_end_time = time.time()


client = OpenAI(
    api_key="38ff4bc74166905637a97ea7685cc0fd:YjIxM2UyZWEwNmVlZWIyY2FjZDEzMTEz",
    base_url = 'https://spark-api-open.xf-yun.com/v1' # 指向讯飞星火的请求地址
)

@func_set_timeout(10)
def gpt_response(model, prompt):
    print("gpt_response")
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt}
        ]
    )
    return completion

gpt_list = []
def gpt_annotation(frame_info_i):
    print("gpt_annotation")
    global gpt_list

    object_info = str(frame_info_i)
    prompt = object_info + prompt_format_benchmark

    gpt_start_time = time.time()
    try:
        completion = gpt_response("generalv3.5", prompt)
        response = completion.choices[0].message.content
        usage = completion.usage
    except:
        print("gpt time running out...")
        return
    
    gpt_end_time = time.time()
    gpt_time_cost = round(gpt_end_time-gpt_start_time, 4)
    gpt_list.append([response, gpt_time_cost, usage])

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
video_name = 'jp2.mp4'

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

fourcc = cv2.VideoWriter_fourcc(*'h264')
output_video = cv2.VideoWriter(output_video_filename, fourcc, 10.0, (img_width, img_height))

'''
detection
'''
detection_info = []
skipped_frame = 6
motion_factor = 10  #60 // skipped_frame

object_list = []
object_alert = []

response_list = []
time_list = []
token_list = []

frame_id = 0
start_time = time.time()
results_list = []
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
        print(f"skip{frame_id}")
        continue

    '''
    yolo inference
    '''
    # results = model.predict(frame)
    results = model(frame)
    results_list.append(results)
    annotated_frame = results[0].plot()

    boxes = results[0].boxes
    xywh = results[0].boxes.xywh
    mask = results[0].masks
    h, w = frame.shape[0:2]

    predictions = boxes.data.cpu().numpy()

    if len(predictions) > 0:
        if len(predictions[0]) == 7:
            boxes = predictions[:, 0:4]
            tracker_id = predictions[:, 4].astype(int)
            classes = predictions[:, 6].astype(int)
            scores = predictions[:, 5]
        else:
            boxes = predictions[:, 0:4]
            tracker_id = np.zeros(len(predictions)).astype(int)
            classes = predictions[:, 5].astype(int)
            scores = predictions[:, 4]
    else:
        continue

    '''
    movements
    '''
    current_frame = [tracker_id, boxes, classes, scores]

    if frame_id > display_start_frame and frame_id % skipped_frame == 0 and skipped_frame > 1:

        frame_info = []
        catrgorized_detections = {'frame_id': frame_id, 'left': [], 'right': [], 'front': [], 'ground': []}

        for pid, box, label, score in zip(tracker_id, boxes, classes, scores):
            x1, y1, x2, y2 = map(int, box)
            class_name = model.names[int(label)]

            if class_name not in object_list:
                object_list.append(class_name)
            
            height = y2 - y1
            width = x2 - x1
            center_x = x1 + (width) // 2
            center_y = y1 + (height) // 2

            height = int(height / h * 100)
            width = int(width / w * 100)
            x_loc = int(center_x / w * 100)
            y_loc = int(center_y /h * 100)

            size = int(height * width / 100)

            if x_loc < 25:
                location = 'left'
            elif x_loc > 75:
                location = 'right'
            elif y_loc < 50:
                location = 'front'
            else:
                location = 'ground'

            info = f"ID: {pid}, class: {class_name}, confidence: {score:.2f}, \
                  center_x: {x_loc}%, center_y: {y_loc}%, object_height:{height}%, object_width:{width}%, size: {size}%"

            catrgorized_detections[location].append(info)
            frame_info.append(info)

            if location in ['ground']:
                object_alert.append([frame_id, location, pid, class_name, score, size])
            if location in ['left', 'right']:
                if size > 20:
                    object_alert.append([frame_id, location, pid, class_name, score, size])
        
        detection_info.append(catrgorized_detections)

        gpt_response_thread = threading.Thread(target=gpt_annotation, args=(catrgorized_detections,))
        gpt_response_thread.start()

        if len(gpt_list) > 0:
            [response, gpt_time_cost, usage] = gpt_list[-1]
            response_list.append(response)
            time_list.append(gpt_time_cost)
        
        try:
            gpt_data = eval(response)
            level = gpt_data['danger_score']
            content = gpt_data['reason']
        except:
            pass

    '''
    Display the frame
    '''
    # Draw lines for the 'H'
    cv2.line(annotated_frame, (left_line_x, 0), (left_line_x, img_height), (0, 255, 0), 10)  # Left vertical line
    cv2.line(annotated_frame, (right_line_x, 0), (right_line_x, img_height), (0, 255, 0), 10)  # Right vertical line
    cv2.line(annotated_frame, (left_line_x, top_line_y), (right_line_x, bottom_line_y), (0, 255, 0),
             10)  # Horizontal line

    if 'level' in locals() and 'content' in locals():
        text_1 = f"Emergency level: {level}"
        text_2 = content
        color = (0, 255 * (1 - level), 255 * level)
        text_x = left_line_x + 10
        text_y = img_height - 100
        cv2.putText(annotated_frame, text_1, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)
        cv2.putText(annotated_frame, text_2, (text_x, text_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    #cv2.imshow('Video Frame', annotated_frame)
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
print('results count: ', len(results_list))
load_time = load_end_time - load_start_time
print(f"load_time: {load_time}")
