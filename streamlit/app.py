# app.py
import streamlit as st
from mtcnn import MTCNN
import cv2
import numpy as np
from PIL import Image
import io
import psutil
import os

def get_memory_usage():
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024  # Convert to MB
    return f"{memory_mb:.2f} MB"

def log_memory(step):
    memory_usage = get_memory_usage()
    st.sidebar.write(f"Memory usage after {step}: {memory_usage}")

# Set page config
st.set_page_config(
    page_title="HowManySwimmers",
    page_icon="🏊",
    layout="centered"
)

# Initialize MTCNN detector
@st.cache_resource
def load_detector():
    detector = MTCNN()
    log_memory("loading MTCNN")
    return detector

detector = load_detector()

def process_image(image):
    log_memory("start of process_image")
    
    # Convert PIL Image to numpy array
    image_np = np.array(image)
    log_memory("after PIL to numpy conversion")
    
    # Convert RGB to BGR (OpenCV format)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # Convert to RGB for MTCNN
    rgb_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    log_memory("before face detection")

    try:
        # Detect faces using MTCNN
        faces = detector.detect_faces(rgb_image)
        log_memory("immediately after face detection")
    finally:
        del rgb_image
        log_memory("after cleaning up detection arrays")

    # Sort faces from left to right
    faces_sorted = sorted(faces, key=lambda x: x['box'][0])

    # Process each face
    for i, face in enumerate(faces_sorted, start=1):
        x, y, width, height = face['box']
        
        # Make sure coordinates are within bounds
        x = max(0, x)
        y = max(0, y)
        width = min(width, image_np.shape[1] - x)
        height = min(height, image_np.shape[0] - y)
        
        # Extract and process face region
        face_region = image_np[y:y+height, x:x+width].copy()
        blurred_face = cv2.GaussianBlur(face_region, (25, 25), 10)
        image_np[y:y+height, x:x+width] = blurred_face
        del face_region, blurred_face

        # Draw rectangle and text
        cv2.rectangle(image_np, (x, y), (x+width, y+height), (0, 255, 0), 2)
        
        font = cv2.FONT_HERSHEY_DUPLEX
        text = str(i)
        text_size = cv2.getTextSize(text, font, 0.5, 1)[0]
        text_x = x + 6
        text_y = y + height - 6
        
        cv2.rectangle(image_np, 
                     (text_x - 2, text_y - text_size[1] - 2),
                     (text_x + text_size[0] + 2, text_y + 2), 
                     (0, 0, 0), -1)
        cv2.putText(image_np, text, (text_x, text_y), font, 0.5, 
                   (255, 255, 255), 1)

    log_memory("after face processing")
    
    # Convert BGR back to RGB
    final_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    del image_np
    log_memory("after final conversion")

    return len(faces), final_image

# Main Streamlit app
def main():
    st.title("HowManySwimmers")
    st.sidebar.title("Memory Usage")
    log_memory("app start")
    
    st.write("""
    Upload an image to count the faces. Each face will be numbered 
    from left to right.
    """)

    uploaded_file = st.file_uploader("Choose an image...", 
                                   type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
        log_memory("after image upload")

        # Process image and display results
        with st.spinner('Processing image...'):
            try:
                num_faces, processed_image = process_image(image)
                log_memory("after processing")
                st.subheader(f"Processed Image: Found {num_faces} Faces")
                st.image(processed_image, use_container_width=True)
                del processed_image
                log_memory("after display")
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                st.error("Memory usage at error: " + get_memory_usage())

if __name__ == '__main__':
    main()