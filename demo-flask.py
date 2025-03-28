from flask import Flask, render_template, Response, request, send_from_directory
import cv2
import time
import os
from threading import Lock

app = Flask(__name__, static_folder='static')
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# 保证上传目录存在
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 使用线程锁保护摄像头对象
camera_lock = Lock()
camera = None

class VideoCamera(object):
    def __init__(self, video_path):
        self.video = cv2.VideoCapture(video_path)
        if not self.video.isOpened():
            raise Exception("Error: Cannot open video file!")

        self.fps = int(self.video.get(cv2.CAP_PROP_FPS))
        self.frame_time = int(1000 / self.fps)  # 计算每帧的间隔时间（毫秒）

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()

        cv2.putText(image, "hello", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2,
                    cv2.LINE_AA)
        ret, jpeg = cv2.imencode('.jpg', image)
        cv2.waitKey(self.frame_time)  # 控制帧率
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
        return "No file selected", 400

    if file:
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(video_path)
        with camera_lock:
            if camera is None:
                del camera
            camera = VideoCamera(video_path)
        return render_template('index.html', video_path=file.filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    # 使用 send_from_directory 来返回视频文件
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=False)

def gen():
    global camera
    prev_time = time.time()
    while True:
        with camera_lock:
            if camera is None:
                break

            frame = camera.get_frame()
            current_time = time.time()
            elapsed_time = current_time - prev_time

            expected_frame_time = 1 / camera.fps
            if elapsed_time < expected_frame_time:
                time.sleep(expected_frame_time - elapsed_time)

            prev_time = time.time()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True, port=50008)
