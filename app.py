import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# 1. การตั้งค่าหน้าเว็บและ CSS ตกแต่ง (STONE LEN Style)
st.set_page_config(page_title="STONE LEN - Rock Classification", layout="wide")

st.markdown("""
    <style>
    /* ตั้งค่าพื้นหลังด้วยรูปแคนยอนจาก Pixabay */
    .stApp {
        background-image: linear-gradient(rgba(0,0,0,0.4), rgba(0,0,0,0.4)), 
                          url("https://pixabay.com/images/download/canyon-1740973_1920.jpg");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    /* ตกแต่งหัวข้อ STONE LEN */
    .main-title {
        color: #dcb799;
        font-size: 70px;
        font-weight: 900;
        text-shadow: 3px 3px 15px rgba(0,0,0,0.8);
        margin-bottom: 0px;
    }

    /* ตกแต่งคำอธิบายภาษาไทย */
    .subtitle {
        color: white;
        font-size: 20px;
        text-shadow: 1px 1px 5px rgba(0,0,0,0.8);
        margin-bottom: 30px;
    }

    /* ตกแต่งกล่องสีขาวสำหรับอัปโหลดรูป */
    .stFileUploader {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 20px;
        padding: 40px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }

    /* ส่วนแสดงผลลัพธ์ */
    .result-box {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 20px;
        margin-top: 20px;
        color: #333;
    }

    /* แถบรายชื่อผู้จัดทำด้านล่าง */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: rgba(45, 62, 51, 0.9);
        color: white;
        text-align: center;
        padding: 10px;
        font-size: 14px;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. ส่วนหัวของหน้าเว็บ
st.markdown('<p class="main-title">STONE LEN</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">ROCK CLASSIFICATION WEBSITE : เว็บไซต์จำแนกประเภทหิน เพื่อการศึกษาทางธรณีวิทยา</p>', unsafe_allow_html=True)

# 3. ฟังก์ชันโหลดโมเดล AI (ใช้ TensorFlow 2.15 ตามที่ตั้งใน requirements.txt)
@st.cache_resource
def load_model():
    # โหลดไฟล์โมเดลที่ชื่อ keras_model.h5
    return tf.keras.models.load_model("keras_model.h5", compile=False)

def load_labels():
    # โหลดรายชื่อหินจากไฟล์ labels.txt
    with open("labels.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

# เรียกใช้งานโมเดลและรายชื่อ
try:
    model = load_model()
    labels = load_labels()
except Exception as e:
    st.error(f"เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")

# 4. ส่วนอัปโหลดและประมวลผล
st.markdown("---")
col1, col2 = st.columns([1.5, 1]) # แบ่งหน้าจอเป็น 2 ฝั่ง

with col1:
    file = st.file_uploader("ลากไฟล์รูปหินมาวางที่นี่ (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if file is not None:
    image = Image.open(file).convert("RGB")
    st.image(image, caption="รูปหินที่คุณอัปโหลด", width=500)
    
    # AI ประมวลผลรูปภาพ
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
                <h2>🔍 ผลการวิเคราะห์</h2>
                <hr>
                <h3>หินชนิดนี้คือ: <b>{labels[index]}</b></h3>
                <p>ความมั่นใจของ AI: <b>{confidence * 100:.2f}%</b></p>
            </div>
        """, unsafe_allow_html=True)

# 5. ส่วนแสดงรายชื่อผู้จัดทำ (Footer)
st.markdown("""
    <div class="footer">
        Creators : Chadaporn Boonnii, Nopanut Channuan, Saranya Changkeb, Phatcharakamon Sodsri
    </div>
    """, unsafe_allow_html=True)
