import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# 1. การตั้งค่าหน้าเว็บและ CSS (เพิ่มคำสั่งซ่อนแถบขาวด้านบน)
st.set_page_config(page_title="STONE LEN - Rock Classification", layout="wide")

st.markdown("""
    <style>
    /* 1. ซ่อนแถบขาวด้านบนสุด (Header) และเมนู */
    header {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    [data-testid="stDecoration"] {display:none;}

    /* 2. พื้นหลัง */
    .stApp {
        background-image: linear-gradient(rgba(0,0,0,0.4), rgba(0,0,0,0.4)), 
                          url("https://images.wallpaperscraft.com/image/single/beach_rocks_stones_136868_3840x2400.jpg");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    /* 3. หัวข้อ STONE LEN (ขยับขึ้นไปชิดขอบบนมากขึ้น) */
    .main-title {
        color: #dcb799 !important;
        font-size: 100px !important;
        font-weight: 900;
        text-shadow: 3px 3px 15px rgba(0,0,0,0.8);
        margin-top: -60px !important; 
        text-align: left;
    }

    /* 4. จัดการกล่องอัปโหลดให้อยู่ตรงกลาง */
    [data-testid="stFileUploader"] {
        width: 350px !important; 
        margin: 0 auto !important;
    }

    [data-testid="stFileUploader"] section {
        background-color: rgba(255, 255, 255, 0.9) !important;
        border-radius: 20px !important;
        padding: 30px !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        text-align: center !important;
    }

    /* 5. เปลี่ยนชื่อปุ่มเป็น Upload file */
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

    .result-box {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 20px;
        color: #333;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }

    .footer-bar {
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
st.markdown('<h1 class="main-title">STONE LEN</h1>', unsafe_allow_html=True)

# ดึงตำแหน่งขึ้นไปหาชื่อ (ใช้ค่าลบที่รุนแรงขึ้น)
st.markdown("""
    <p style="color: white; 
              font-size: 20px; 
              text-shadow: 1px 1px 5px rgba(0,0,0,0.8);
              position: relative; 
              top: -45px; 
              left: 10px;
              margin-bottom: -40px;">
        ROCK CLASSIFICATION WEBSITE : เว็บไซต์จำแนกประเภทหิน เพื่อการศึกษาทางธรณีวิทยา
    </p>
    """, unsafe_allow_html=True)

# 3. โหลดโมเดล
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("keras_model.h5", compile=False)

def load_labels():
    with open("labels.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

model = load_model()
labels = load_labels()

# 4. ส่วนอัปโหลด
st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if file is not None:
    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    image = Image.open(file).convert("RGB")
    with col1:
        st.image(image, caption="รูปที่อัปโหลด", use_container_width=True)
    
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
    <div class="footer-bar">
        Creators : Chadaporn Boonnii, Nopphanat Junnunl, Saranya Changkeb, Phatcharakamon Sodsri
</div>
    """, unsafe_allow_html=True)
