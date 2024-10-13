import PIL
import streamlit as st
from ultralytics import YOLO
import cv2
import requests
import numpy as np
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
        # Konversi gambar PIL ke format OpenCV
        image_cv = np.array(image.convert('RGB'))
        image_cv = image_cv[:, :, ::-1].copy()

        # Prediksi
        res = model.predict(image_cv, conf=confidence)
        boxes = res[0].boxes

        # Warna acak untuk setiap bounding box
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Koordinat bounding box
            color = tuple(np.random.randint(0, 255, 3).tolist())  # Warna acak
            cv2.rectangle(image_cv, (x1, y1), (x2, y2), color, 2)  # Gambar bounding box
            cv2.putText(image_cv, "Jeruk", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Tampilkan gambar dengan bounding box berwarna
        res_plotted = image_cv[:, :, ::-1]  # Konversi kembali ke format RGB untuk ditampilkan
        with col2:
            st.image(res_plotted, caption='Gambar Terdeteksi', use_column_width=True)
            with st.expander("Hasil Deteksi"):
                for box in boxes:
                    st.write(box.xywh)

# Peringatan untuk Cloud Streamlit
if use_camera:
    st.warning("Akses kamera tidak didukung di Cloud Streamlit. Unggah gambar atau gunakan URL gambar untuk dideteksi.")
