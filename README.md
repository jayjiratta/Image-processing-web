# Image Processing Web (Streamlit)

A web application for real-time webcam and image URL processing with interactive controls, built using Streamlit, OpenCV, and Matplotlib.

## Features
 - Realtime webcam display
 - Support for image URL (JPG/PNG) as input (no video stream support)
 - Image processing options:
   - Gray Scale (with brightness adjustment)
   - Invert (with adjustable intensity)
   - Blur (adjustable kernel size)
   - Canny Edge Detection (adjustable thresholds)
   - Sobel Edge Detection (adjustable kernel size)
   - Prewitt Edge Detection (adjustable kernel size)
 - Show both original and processed images side by side
 - Show pixel intensity histogram:
   - Combined RGB histogram
   - Separate B, G, R channel histograms (subplot)
 - Start/Stop camera stream with button

## How to Run
1. Install dependencies:
   ```bash
   pip install streamlit opencv-python numpy pillow matplotlib
   ```
2. Run the app:
   ```bash
   streamlit run main.py
   ```
3. Open the provided local URL in your browser.

## Usage
- Select camera source: Webcam or URL
- If URL, direct image link (JPG/PNG)
- Choose image processing method and adjust parameters in the sidebar
- View original and processed images, and histograms in real time
- Use Stop/Start buttons to control the stream
