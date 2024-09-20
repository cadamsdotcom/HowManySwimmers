# app.py
from flask import Flask, render_template, request, jsonify
from mtcnn import MTCNN
import cv2
import numpy as np
import base64

app = Flask(__name__)

# Initialize MTCNN detector
detector = MTCNN()

def enhanced_upscale(image, scale_factor=2):
    # Upscale using INTER_CUBIC
    upscaled = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

    # Apply slight Gaussian blur
    blurred = cv2.GaussianBlur(upscaled, (0, 0), 1)

    # Sharpen the image
    sharpened = cv2.addWeighted(upscaled, 1.5, blurred, -0.5, 0)

    return sharpened

def process_image(image):
    # Read image file
    image_np = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)

    # Upscale the image using our enhanced method
    upscaled = enhanced_upscale(image_np, scale_factor=2)

    # Convert the image from BGR color (which OpenCV uses) to RGB color
    rgb_image = cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB)

    # Detect faces using MTCNN
    faces = detector.detect_faces(rgb_image)

    # Sort faces from left to right
    faces_sorted = sorted(faces, key=lambda x: x['box'][0])

    # Draw rectangles around the faces and add numbers
    for i, face in enumerate(faces_sorted, start=1):
        x, y, width, height = face['box']

        # Draw the rectangle
        cv2.rectangle(upscaled, (x, y), (x+width, y+height), (0, 255, 0), 2)

        # Add the number
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(upscaled, str(i), (x + 6, y + height - 6), font, 0.5, (255, 255, 255), 1)

    # Downscale the image back to original size for display
    image_np = cv2.resize(upscaled, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_AREA)

    # Convert the image to base64 for sending to frontend
    _, buffer = cv2.imencode('.jpg', image_np)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return len(faces), img_base64

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