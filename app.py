import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
import io

st.title("Image Processing App")
st.markdown("---")

st.sidebar.header("Parameters")

width       = st.sidebar.number_input("Width (px)",  min_value=100, max_value=3840, value=1080, step=10)
height      = st.sidebar.number_input("Height (px)", min_value=100, max_value=2160, value=720,  step=10)
blur_kernel = st.sidebar.slider("Blur Amount", min_value=1, max_value=51, value=11, step=2)
edge_low    = st.sidebar.slider("Edge Lower Threshold", min_value=0, max_value=255, value=100)
edge_high   = st.sidebar.slider("Edge Upper Threshold", min_value=0, max_value=255, value=200)

# Helper: convert image array to downloadable bytes 
def image_to_bytes(img_array, filename="result.png"):
    pil_img = Image.fromarray(img_array)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return buf

# Image Source Selection 
st.subheader("Select Image Source")
option = st.radio("", ["Browse (Upload)", "Sample (from images folder)"], horizontal=True)

img_rgb = None

# Option 1: Upload 
if option == "Browse (Upload)":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])
    if uploaded_file is not None:
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Option 2: Sample folder 
elif option == "Sample (from images folder)":
    IMAGE_FOLDER = "images"
    image_files  = []
    if os.path.exists(IMAGE_FOLDER):
        image_files = [
            f for f in os.listdir(IMAGE_FOLDER)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ]

    if not image_files:
        st.error("No images found in the images/ folder.")
        st.stop()

    selected_file = st.selectbox("Select an image", image_files)
    image_path    = os.path.join(IMAGE_FOLDER, selected_file)
    img_bgr       = cv2.imread(image_path)
    img_rgb       = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Stop if no image loaded
if img_rgb is None:
    st.info("Please select or upload an image to continue.")
    st.stop()

# Resize
img_resized = cv2.resize(img_rgb, (width, height))

# Show resized image
st.markdown("---")
st.subheader("Resized Image")
st.image(img_resized, caption=f"{width} x {height} px", use_container_width=True)

# Download original resized image
st.download_button(
    label     = "Download Resized Image",
    data      = image_to_bytes(img_resized),
    file_name = "resized_image.png",
    mime      = "image/png"
)

st.markdown("---")
st.subheader("Apply Effect")

# Buttons
col1, col2, col3 = st.columns(3)
gray_btn = col1.button("Gray", use_container_width=True)
blur_btn = col2.button("Blur", use_container_width=True)
edge_btn = col3.button("Edge", use_container_width=True)

# Process, Show & Download 
result        = None
result_name   = "result.png"
result_label  = "Download Result"

if gray_btn:
    result       = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    result_name  = "gray_image.png"
    result_label = "Download Grayscale Image"
    st.image(result, caption="Grayscale Image", use_container_width=True, clamp=True)

elif blur_btn:
    k            = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
    result       = cv2.GaussianBlur(img_resized, (k, k), 0)
    result_name  = "blurred_image.png"
    result_label = "Download Blurred Image"
    st.image(result, caption=f"Blurred Image (kernel={k})", use_container_width=True)

elif edge_btn:
    gray_temp    = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    result       = cv2.Canny(gray_temp, edge_low, edge_high)
    result_name  = "edge_image.png"
    result_label = "Download Edge Image"
    st.image(result, caption="Edge Detection", use_container_width=True, clamp=True)

else:
    st.info("Click a button above to apply an effect.")

# download button only after effect is applied
if result is not None:
    st.download_button(
        label     = result_label,
        data      = image_to_bytes(result),
        file_name = result_name,
        mime      = "image/png"
    )