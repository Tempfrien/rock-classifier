import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
from style_config import apply_custom_style  # <--- เรียกใช้ไฟล์ UI ที่เราแยกไว้

# 1. ตั้งค่าหน้าเว็บและดึงสไตล์มาจากไฟล์ style_config.py
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
    # --- Logic การประมวลผล AI ---
    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    image = Image.open(file).convert("RGB")
    with col1:
        st.image(image, caption="รูปที่อัปโหลด", use_container_width=True)
    
    # ... (โค้ดประมวลผลเหมือนเดิม) ...
    # สรุปผลแสดงที่ col2

# 5. Footer
st.markdown('<div class="footer-bar">Creators : Chadaporn Boonnii, Nopphanat Junnunl...</div>', unsafe_allow_html=True)
