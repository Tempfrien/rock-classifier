import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

st.set_page_config(page_title="Rock Classifier", page_icon="🪨")
st.title("Stone Len")
st.write("อัปโหลดรูปภาพที่ช่องด้านล่างเพื่อตรวจสอบชนิดของหิน")
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("keras_model.h5", compile=False)

def load_labels():
    with open("labels.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

model = load_model()
labels = load_labels()

file = st.file_uploader("เลือกรูปภาพหิน...", type=["jpg", "jpeg", "png"])

if file is not None:
    image = Image.open(file).convert("RGB")
    st.image(image, caption="รูปที่อัปโหลด", use_container_width=True)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image)
    normalized_img = (img_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_img
    prediction = model.predict(data)
    index = np.argmax(prediction)
    st.success(f"นี่คือ: {labels[index]}")
    st.info(f"ความมั่นใจ: {prediction[0][index] * 100:.2f}%")
