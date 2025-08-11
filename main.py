import streamlit as st
import cv2
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Realtime Webcam Image Processing")

# Sidebar controls
st.sidebar.header("Image Processing Controls")
processing = st.sidebar.selectbox(
    "Choose Processing",
    ["None", "Gray Scale", "Invert", "Blur", "Canny Edge", "Sobel Edge", "Prewitt Edge"]
)

if processing == "Gray Scale":
    gray_brightness = st.sidebar.slider("Gray Brightness", -100, 100, 0)
elif processing == "Blur":
    blur_ksize = st.sidebar.slider("Blur Kernel Size", 1, 31, 5, step=2)
elif processing == "Canny Edge":
    canny_th1 = st.sidebar.slider("Canny Threshold 1", 0, 255, 100)
    canny_th2 = st.sidebar.slider("Canny Threshold 2", 0, 255, 200)
elif processing == "Sobel Edge":
    sobel_ksize = st.sidebar.slider("Sobel Kernel Size", 1, 7, 3, step=2)
elif processing == "Prewitt Edge":
    prewitt_ksize = st.sidebar.slider("Prewitt Kernel Size (odd)", 1, 7, 3, step=2)

# Camera selection
source = st.sidebar.radio("Camera Source", ["Webcam", "URL"])
if source == "URL":
    cam_url = st.sidebar.text_input("Stream URL", "http://...")
else:
    cam_url = 0

# State management
if "stop" not in st.session_state:
    st.session_state.stop = False
if "frame" not in st.session_state:
    st.session_state.frame = None

col1, col2 = st.columns(2)
start_btn = st.button("Start Camera")
stop_btn = st.button("Stop Camera")

def process_image(img):
    if processing == "Gray Scale":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if 'gray_brightness' in globals():
            # ปรับ brightness
            gray = np.clip(gray.astype(np.int16) + gray_brightness, 0, 255).astype(np.uint8)
        return gray
    elif processing == "Invert":
        return cv2.bitwise_not(img)
    elif processing == "Blur":
        return cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)
    elif processing == "Canny Edge":
        return cv2.Canny(img, canny_th1, canny_th2)
    elif processing == "Sobel Edge":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
        sobel = cv2.magnitude(sobelx, sobely)
        sobel = np.uint8(np.clip(sobel, 0, 255))
        return sobel
    elif processing == "Prewitt Edge":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # สร้าง kernel ขนาด prewitt_ksize แบบ odd
        k = prewitt_ksize if 'prewitt_ksize' in globals() else 3
        kernelx = np.zeros((k, k), dtype=np.float32)
        kernelx[:, 0] = 1
        kernelx[:, -1] = -1
        kernely = np.zeros((k, k), dtype=np.float32)
        kernely[0, :] = 1
        kernely[-1, :] = -1
        prewittx = cv2.filter2D(gray, -1, kernelx)
        prewitty = cv2.filter2D(gray, -1, kernely)
        prewitt = cv2.magnitude(prewittx.astype(np.float32), prewitty.astype(np.float32))
        prewitt = np.uint8(np.clip(prewitt, 0, 255))
        return prewitt
    else:
        return img

def plot_histogram(img):
    if len(img.shape) == 2:
        fig, ax = plt.subplots()
        ax.hist(img.ravel(), bins=256, color='gray')
        ax.set_title("Gray Pixel Intensity Histogram")
        st.pyplot(fig)
    else:
        fig1, ax1 = plt.subplots(figsize=(6, 3))
        colors = ('b','g','r')
        for i, col in enumerate(colors):
            ax1.hist(img[:,:,i].ravel(), bins=256, color=col, alpha=0.5)
        ax1.set_title("RGB Pixel Intensity Histogram")
        st.pyplot(fig1)

        fig2, axs = plt.subplots(3, 1, figsize=(6, 8))
        titles = ('Blue Channel', 'Green Channel', 'Red Channel')
        for i, (col, title) in enumerate(zip(colors, titles)):
            axs[i].hist(img[:,:,i].ravel(), bins=256, color=col)
            axs[i].set_title(title)
        plt.tight_layout()
        st.pyplot(fig2)

if start_btn:
    st.session_state.stop = False

if stop_btn:
    st.session_state.stop = True

if not st.session_state.stop and start_btn:
    cap = cv2.VideoCapture(cam_url)
    frame_placeholder = col1.empty()
    proc_placeholder = col2.empty()
    hist_placeholder = st.empty()
    while cap.isOpened() and not st.session_state.stop:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to grab frame.")
            break
        st.session_state.frame = frame.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", caption="Webcam (RGB)")
        proc_img = process_image(frame)
        if len(proc_img.shape) == 2:
            proc_img_disp = proc_img
        else:
            proc_img_disp = cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB)
        proc_placeholder.image(proc_img_disp, caption="Processed Image")
        hist_placeholder.empty()
        plot_histogram(proc_img)
        # ใช้ปุ่ม stop_btn ที่สร้างไว้ด้านบนแทน ไม่ต้องสร้างปุ่มซ้ำใน loop
        if st.session_state.stop:
            break
    cap.release()
elif st.session_state.frame is not None:
    frame = st.session_state.frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    col1.image(frame_rgb, channels="RGB", caption="Webcam (RGB)")
    proc_img = process_image(frame)
    if len(proc_img.shape) == 2:
        proc_img_disp = proc_img
    else:
        proc_img_disp = cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB)
    col2.image(proc_img_disp, caption="Processed Image")
    plot_histogram(proc_img)
else:
    st.info("Press 'Start Camera' to begin.")