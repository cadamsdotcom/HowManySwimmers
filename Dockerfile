FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y cmake build-essential ffmpeg libsm6 libxext6

RUN pip install flask
RUN pip install torch
RUN pip install torchvision
RUN pip install opencv-contrib-python
RUN pip install numpy
RUN pip install pandas
RUN pip install requests
RUN pip install yolov5
RUN pip install face_recognition
RUN pip install dlib
RUN pip install MTCNN
RUN pip install tensorflow
RUN pip install gunicorn
#COPY requirements.txt .
#RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY templates/* ./templates/

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]