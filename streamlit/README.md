# HowManySwimmers: Face Detection App

This Streamlit application detects faces in images and applies a blur effect while numbering each face from left to right.

It's similar to the app in the root directory but doesn't do upscaling, has a bit of memory debugging, and crashes on Streamlit community cloud when it goes over 1GB of memory (it needs about 15GB for a 2.5MB image, even without upscaling!)

## Features
- Face detection using MTCNN
- Automatic face blurring
- Face numbering
- Image enhancement
- Responsive web interface

## Local Development Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```

## Deploying to Streamlit Community Cloud

1. Create a GitHub repository and push your code:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin <your-repo-url>
   git push -u origin main
   ```

2. Sign up for Streamlit Community Cloud at https://streamlit.io/cloud

3. In the Streamlit dashboard:
   - Click "New app"
   - Select your repository
   - Select main branch
   - Set the path to app.py
   - Click "Deploy"

4. Your app will be available at https://<your-app-name>.streamlit.app

## Important Notes

- The app uses CPU-based face detection. For better performance, consider using GPU-enabled instances if available.
- The MTCNN model is cached using @st.cache_resource to improve performance.
- Images are processed in memory and are not stored permanently.
- The app uses opencv-python-headless instead of opencv-contrib-python for cloud compatibility.

## Troubleshooting

If you encounter deployment issues:
1. Ensure all dependencies are in requirements.txt
2. Check that you're using opencv-python-headless instead of opencv-contrib-python
3. Verify that your Python version is compatible with all dependencies
4. Monitor the deployment logs in Streamlit Cloud for specific errors