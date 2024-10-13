import PIL
import streamlit as st
from ultralytics import YOLO
import cv2
import requests
from io import BytesIO

model_path = 'best.pt'

st.set_page_config(
    page_title="Deteksi Jeruk",
    page_icon="🍊",
    layout="wide", 
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.header("Konfigurasi Gambar") 
    source_img = st.file_uploader("Unggah gambar...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))
    image_url = st.text_input("Atau masukkan URL gambar:")
    confidence = float(st.slider("Pilih Tingkat Kepercayaan Model", 0, 100, 40)) / 100
    use_camera = st.checkbox("Gunakan Kamera?")

st.title("Deteksi Buah Jeruk 🍊")
col1, col2 = st.columns(2)

try:
    model = YOLO(model_path)
except Exception as ex:
    st.error(f"Tidak dapat memuat model. Periksa jalur yang ditentukan: {model_path}")
    st.error(ex)

image = None

# Via File
if source_img:
    image = PIL.Image.open(source_img)

# Via URL
elif image_url:
    try:
        response = requests.get(image_url)
        image = PIL.Image.open(BytesIO(response.content))
    except Exception as ex:
        st.error("Tidak dapat memuat gambar dari URL yang diberikan.")
        st.error(ex)

# Proses
if image:
    with col1:
        st.image(image, caption="Gambar yang Diunggah", use_column_width=True)

    if st.sidebar.button('Deteksi Objek'):
        res = model.predict(image, conf=confidence)
        boxes = res[0].boxes
        res_plotted = res[0].plot()[:, :, ::-1]
        with col2:
            st.image(res_plotted, caption='Gambar Terdeteksi', use_column_width=True)
            with st.expander("Hasil Deteksi"):
                for box in boxes:
                    st.write(box.xywh)

# Via Camera
if use_camera:
    cap = cv2.VideoCapture(0)
    st.write("Deteksi Kamera Real-time Diaktifkan")
    camera_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Gagal menangkap frame video.")
            break

        results = model.predict(frame, conf=confidence)
        frame = results[0].plot()[:, :, ::-1]

        camera_placeholder.image(frame, channels="RGB", use_column_width=True)

    cap.release()
    # cv2.destroyAllWindows()
