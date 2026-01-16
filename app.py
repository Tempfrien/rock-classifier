from style_config import apply_custom_style
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np


# 1. ตั้งค่าหน้าเว็บและดึงสไตล์
st.set_page_config(page_title="STONE LEN - Rock Classification", layout="wide")
apply_custom_style()

# 2. แสดงผล UI (ส่วนหัว)
st.markdown('<h1 class="main-title">STONE LEN</h1>', unsafe_allow_html=True)
st.markdown("""
    <p style="color: white; font-size: 20px; text-shadow: 1px 1px 5px rgba(0,0,0,0.8);
              position: relative; top: -45px; left: 10px; margin-bottom: -40px;">
        ROCK CLASSIFICATION WEBSITE : เว็บไซต์จำแนกประเภทหิน เพื่อการศึกษาทางธรณีวิทยา
    </p>
    """, unsafe_allow_html=True)

# 3. Logic การทำงาน (AI & Model)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("keras_model.h5", compile=False)

def load_labels():
    with open("labels.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

model = load_model()
labels = load_labels()

# 4. ส่วนรับข้อมูลภาพ
st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if file is not None:
    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    image = Image.open(file).convert("RGB")
    
    with col1:
        st.image(image, caption="รูปที่อัปโหลด", use_container_width=True)
    
    # --- เริ่มการประมวลผล AI ---
    size = (224, 224)
    image_processed = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image_processed)
    normalized_img = (img_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_img
    
    prediction = model.predict(data)
    index = np.argmax(prediction)
    confidence = prediction[0][index]
    
    with col2:
        st.markdown(f"""
            <div class="result-box">
                <h2 style='text-align:center;'>🔍 ผลการวิเคราะห์</h2>
                <hr>
                <p style='font-size:20px;'>หินชนิดนี้คือ: <b style='color:#dcb799;'>{labels[index]}</b></p>
                <p style='font-size:18px;'>ความแม่นยำ: <b>{confidence * 100:.2f}%</b></p>
            </div>
        """, unsafe_allow_html=True)

# 5. Footer
st.markdown('<div class="footer-bar">Creators : Chadaporn Boonnii, Nopphanat Junnunl, Saranya Changkeb, Phatcharakamon Sodsri</div>', unsafe_allow_html=True)
