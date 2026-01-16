import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# 1. การตั้งค่าหน้าเว็บและ CSS เพื่อความสวยงาม
st.set_page_config(page_title="STONE LEN - Rock Classification", layout="wide")

st.markdown("""
    <style>
    /* ใส่ภาพพื้นหลัง */
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.5), rgba(0,0,0,0.5)), 
                    url("https://pixabay.com/images/download/canyon-1740973_1920.jpg");
        background-size: cover;
    }
    /* ปรับแต่งส่วนหัว (Title) */
    .main-title {
        font-size: 60px !important;
        font-weight: bold;
        color: #FAD02C; /* สีทองแบบในรูป */
        text-shadow: 2px 2px 4px #000000;
        margin-bottom: 0px;
    }
    /* ปรับแต่งกร่องอัปโหลดรูป */
    .stFileUploader {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        padding: 20px;
    }
    /* ส่วน Footer ด้านล่าง */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #2D3E33;
        color: white;
        text-align: center;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. ส่วนหัวเว็บไซต์ (Header) เหมือนในรูป
st.markdown('<p class="main-title">STONE LEN</p>', unsafe_allow_html=True)
st.write("ROCK CLASSIFICATION WEBSITE : เว็บไซต์จำแนกประเภทหิน เพื่อการศึกษาทางธรณีวิทยา")

# 3. โหลดโมเดล AI
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("keras_model.h5", compile=False)

def load_labels():
    with open("labels.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

model = load_model()
labels = load_labels()

# 4. ส่วนอัปโหลดรูปภาพ
st.markdown("---")
col1, col2 = st.columns([2, 1])

with col1:
    file = st.file_uploader("Drag and drop file here (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if file is not None:
    image = Image.open(file).convert("RGB")
    st.image(image, caption="รูปหินที่เลือก", width=400)
    
    # ประมวลผล AI
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image)
    normalized_img = (img_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_img
    
    prediction = model.predict(data)
    index = np.argmax(prediction)
    
    with col2:
        st.markdown(f"### ผลการวิเคราะห์:")
        st.subheader(f"ชนิดหิน: {labels[index]}")
        st.write(f"ความแม่นยำ: {prediction[0][index] * 100:.2f}%")

# 5. ส่วนแสดงรายชื่อผู้จัดทำ (Footer)
st.markdown("""
    <div class="footer">
        Creators : Chadaporn Boonnii, Nopanut Channuan, Saranya Changkeb, Phatcharakamon Sodsri
    </div>
    """, unsafe_allow_html=True)
