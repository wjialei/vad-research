# -*- coding: utf-8 -*-
"""
Created on Sun Mar 07 19:48:35 2024

@author: MaxGr
"""

import os
import cv2
import time
import numpy as np
from datetime import datetime
from ultralytics import YOLO
import torch
from openai import OpenAI
from func_timeout import func_set_timeout
import threading
from threading import Lock
from flask import Flask, render_template, Response, request, send_from_directory
from werkzeug.utils import secure_filename
import subprocess

# Flask settings
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'flv'}
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def convert_video(input_path, output_path):
    command = [
        'ffmpeg',
        '-i', input_path,
        '-c:v', 'libx264',
        '-c:a', 'aac',
        '-strict', 'experimental',
        output_path
    ]
    subprocess.run(command, check=True)

# 使用线程锁保护摄像头对象
camera_lock = Lock()
camera = None

class VideoCamera(object):
    def __init__(self, video_path):
        self.video = cv2.VideoCapture(video_path)
        if not self.video.isOpened():
            raise Exception("Cannot open video file")

        self.fps = int(self.video.get(cv2.CAP_PROP_FPS))
        self.frame_time = int(1000 / self.fps)
        # model
        weight_file = 'yolov8x-worldv2.pt'
        self.model = YOLO(weight_file)
        self.model.set_classes(CLASSES)

    def __del__(self):
        self.video.release()

    def get_frame(self, frame_id):
        # Check if the video file opened successfully
        ret, frame = self.video.read()
        if not ret:
            self.video.release()
            return None

        # Get frame dimensions
        img_height, img_width = frame.shape[:2]

        # Calculate 'H' segmentation lines
        left_line_x = img_width // 4
        right_line_x = img_width * 3 // 4
        top_line_y = img_height // 2
        bottom_line_y = img_height // 2

        cv2.putText(frame, f'{frame_id - 1}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color[mark_danger], 2,
                    cv2.LINE_AA)

        '''
        yolo inference
        '''
        results = self.model(frame)
        annotated_frame = results[0].plot()
        boxes = results[0].boxes
        xywh = results[0].boxes.xywh
        mask = results[0].masks
        h, w = frame.shape[0:2]

        # 初始化默认值
        tracker_id = np.array([], dtype=int)
        boxes_np = np.array([])
        classes = np.array([], dtype=int)
        scores = np.array([])

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
        '''
        movements
        '''
        current_frame = [tracker_id, boxes, classes, scores]

        if frame_id > display_start_frame and frame_id % skipped_frame == 0 and skipped_frame > 1:

            frame_info = []
            catrgorized_detections = {'frame_id': frame_id, 'left': [], 'right': [], 'front': [], 'ground': []}

            for pid, box, label, score in zip(tracker_id, boxes, classes, scores):
                x1, y1, x2, y2 = map(int, box)
                class_name = self.model.names[int(label)]

                if class_name not in object_list:
                    object_list.append(class_name)

                height = y2 - y1
                width = x2 - x1
                center_x = x1 + (width) // 2
                center_y = y1 + (height) // 2

                height = int(height / h * 100)
                width = int(width / w * 100)
                x_loc = int(center_x / w * 100)
                y_loc = int(center_y / h * 100)

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

        ret2, jpeg = cv2.imencode('.jpg', annotated_frame)
        cv2.waitKey(self.frame_time)
        return jpeg.tobytes()

@app.route('/')
def index():
    return render_template('index.html', video_path=None)


@app.route('/upload', methods=['POST'])
def upload_video():
    global camera
    if 'video_file' not in request.files:
        return "No video file", 400
    file = request.files['video_file']
    if file.filename == '':
        return "Invalid filename", 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)
        with camera_lock:
            if camera is None:
                del camera
            camera = VideoCamera(video_path)
        return render_template('index.html', video_path=file.filename)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=False)


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
TORCH_CUDA_ARCH_LIST = "8.6"
current_datetime = datetime.now()
date_time_string = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

print('torch.version: ', torch.__version__)
print('torch.version.cuda: ', torch.version.cuda)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# objects
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

# prompts
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


client = OpenAI(
    api_key="38ff4bc74166905637a97ea7685cc0fd:YjIxM2UyZWEwNmVlZWIyY2FjZDEzMTEz",
    base_url='https://spark-api-open.xf-yun.com/v1'  # 指向讯飞星火的请求地址
)


@func_set_timeout(10)
def gpt_response(language_model, prompt):
    print("gpt_response")
    completion = client.chat.completions.create(
        model=language_model,
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
    gpt_time_cost = round(gpt_end_time - gpt_start_time, 4)
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


def gen():
    global camera
    prev_time = time.time()
    frame_id = 0
    while True:
        with camera_lock:
            if camera is None:
                break

            frame_id += 1
            print('Current frame: ', frame_id)

            frame = camera.get_frame(frame_id)
            if frame is None:
                camera = None
                break

            current_time = time.time()
            elapsed_time = current_time - prev_time

            if frame_id < display_start_frame:
                continue
            if frame_id > display_until_frame:
                break
            if frame_id % 5 in [1, 2, 3, 4]:
                continue
            expected_frame_time = 1 / camera.fps
            if elapsed_time < expected_frame_time:
                time.sleep(expected_frame_time - elapsed_time)

            prev_time = time.time()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True, port=50001)

