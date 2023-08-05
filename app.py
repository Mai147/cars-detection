from flask import Flask, render_template, request, Response
import cv2
from pathlib import Path
import os
from ultralytics import YOLO
from moviepy.editor import VideoFileClip

app = Flask(__name__)

model = YOLO('./models/best.pt')

def compress_video(input_path, output_path, width):
    video = VideoFileClip(input_path)
    compressed_video = video.resize(width=width)  # Set the desired width (height will be adjusted automatically)
    compressed_video.write_videofile(output_path)
    video.close()
    compressed_video.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if not os.path.exists('uploads'):
            os.makedirs('uploads')
        video_path = Path('uploads/') / file.filename
        file.save(str(video_path))

        video = cv2.VideoCapture(str(video_path))

        frames = []

        fps = video.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        input_video_path = 'result_video.mp4'
        output_video_path = 'result_video_compress.mp4'
        video_writer = cv2.VideoWriter(input_video_path, fourcc, fps, (width, height))

        while True:
            success, frame = video.read()
            if (success == False):
                break
            results = model(frame, stream=True)
            for r in results:
                boxes=r.boxes
                for box in boxes:
                    x1,y1,x2,y2=box.xyxy[0]
                    x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,255),2)

            frames.append(frame)
        
        for frame in frames:
            video_writer.write(frame)

        video_writer.release()

        compress_video(input_video_path, output_video_path, width)

        with open(output_video_path, 'rb') as f:
            video_data = f.read()

        # Remove the temporary video file
        Path('result_video.mp4').unlink()

        video.release()

        os.remove(video_path)
        os.remove(output_video_path)

        return Response(video_data, mimetype='video/mp4')

if __name__ == "__main__":
    app.run(debug=True)