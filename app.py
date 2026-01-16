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
                          url("data:image/jpeg;base64,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");
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
