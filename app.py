# app.py
from flask import Flask, render_template, request, jsonify
import face_recognition
import cv2
import numpy as np
import base64

app = Flask(__name__)

def process_image(image):
    # Read image file
    image_np = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Convert the image from BGR color (which OpenCV uses) to RGB color
    rgb_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    
    # Find all face locations in the image
    face_locations = face_recognition.face_locations(rgb_image)
    
    # Sort face locations from left to right
    face_locations_sorted = sorted(face_locations, key=lambda x: x[3])  # x[3] is the left coordinate
    
    # Draw rectangles around the faces and add numbers
    for i, (top, right, bottom, left) in enumerate(face_locations_sorted, start=1):
        # Draw the rectangle
        cv2.rectangle(image_np, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # Add the number
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image_np, str(i), (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
    
    # Convert the image to base64 for sending to frontend
    _, buffer = cv2.imencode('.jpg', image_np)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return len(face_locations), img_base64

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        if file:
            try:
                num_faces, img_base64 = process_image(file)
                return jsonify({'count': int(num_faces), 'image': img_base64})
            except Exception as e:
                return jsonify({'error': str(e)})
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)