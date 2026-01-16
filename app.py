import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# 1. การตั้งค่าหน้าเว็บและ CSS ตกแต่ง
st.set_page_config(page_title="STONE LEN - Rock Classification", layout="wide")

st.markdown("""
    <style>
    /* พื้นหลัง */
    .stApp {
        background-image: linear-gradient(rgba(0,0,0,0.4), rgba(0,0,0,0.4)), 
                          url("https://images.wallpaperscraft.com/image/single/beach_rocks_stones_136868_3840x2400.jpg");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    /* หัวข้อ STONE LEN */
    .main-title {
        color: #dcb799 !important;
        font-size: 100px !important;
        font-weight: 900;
        text-shadow: 3px 3px 15px rgba(0,0,0,0.8);
        margin-bottom: 0px;
        text-align: left;
    }

    .subtitle {
        color: white;
        font-size: 20px;
        text-shadow: 1px 1px 5px rgba(0,0,0,0.8);
        margin-bottom: 30px;
        text-align: left;
    }

    /* จัดการกล่องอัปโหลดให้อยู่ตรงกลางหน้าจอและจัดสิ่งที่อยู่ข้างในให้ตรงกลาง */
    [data-testid="stFileUploader"] {
        width: 310px !important; /* เพิ่มความกว้างให้สมดุล */
        margin: 0 auto !important;
    }

    [data-testid="stFileUploader"] section {
        background-color: rgba(255, 255, 255, 0.9) !important;
        border-radius: 20px !important;
        padding: 40px !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important; /* จัดปุ่มให้อยู่กลาง */
        text-align: center !important;
    }

    /* เปลี่ยนชื่อปุ่ม Browse files เป็น Upload file และจัดให้อยู่กลาง */
    button[kind="secondary"] {
        font-size: 0 !important;
        border-radius: 30px !important;
        padding: 10px 30px !important;
        background-color: white !important;
        border: 1px solid #ccc !important;
        display: block !important;
        margin: 0 auto !important;
    }
    button[kind="secondary"]::after {
        content: "Upload file";
        font-size: 16px !important;
        color: #333;
    }

    /* ส่วนแสดงผลลัพธ์ */
    .result-box {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 20px;
        color: #333;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }

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
        z-index: 999;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. ส่วนหัวของหน้าเว็บ
st.markdown('<p class="main-title">STONE LEN</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">ROCK CLASSIFICATION WEBSITE : เว็บไซต์จำแนกประเภทหิน เพื่อการศึกษาทางธรณีวิทยา</p>', unsafe_allow_html=True)

# 3. ฟังก์ชันโหลดโมเดล
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("keras_model.h5", compile=False)

def load_labels():
    with open("labels.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

try:
    model = load_model()
    labels = load_labels()
except Exception as e:
    st.error(f"Error: {e}")

# 4. ส่วนอัปโหลด
st.markdown("<br>", unsafe_allow_html=True)
file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if file is not None:
    # เมื่ออัปโหลดแล้ว ค่อยแบ่งเป็น 2 คอลัมน์เพื่อโชว์รูปและผลลัพธ์
    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    
    image = Image.open(file).convert("RGB")
    with col1:
        st.image(image, caption="รูปที่อัปโหลด", use_container_width=True)
    
    # AI ประมวลผล
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
                <p style='font-size:18px;'>ความมั่นใจ: <b>{confidence * 100:.2f}%</b></p>
            </div>
        """, unsafe_allow_html=True)

# 5. Footer
st.markdown(f"""
    <div class="footer">
        Creators : Chadaporn Boonnii, Nopphanat Junnunl, Saranya Changkeb, Phatcharakamon Sodsri
    </div>
    """, unsafe_allow_html=True)
