import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# 1. การตั้งค่าหน้าเว็บและ CSS เพื่อความสวยงาม
st.set_page_config(page_title="STONE LEN - Rock Classification", layout="wide")

st.markdown("""
    <style>
    /* ตกแต่ง Title 'STONE LEN' */
    .main-title {
        color: #dcb799; /* สีเหลืองทอง */
        font-size: 1000px;
        font-weight: 900;
        text-align: left;
        margin-top: -50px;
    }

    /* ตกแต่งกล่องอัปโหลดรูปภาพ */
    .stFileUploader {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 25px;
        padding: 40px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    
    /* ปรับแต่งส่วนคำอธิบาย (Subtitle) */
    .subtitle {
        color: white;
        font-size: 18px;
        text-shadow: 1px 1px 5px rgba(0,0,0,0.5);
    }
    </style>
    """, unsafe_allow_html=True)

# ส่วนหัวของหน้าเว็บ
st.markdown('<p class="main-title">STONE LEN</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">ROCK CLASSIFICATION WEBSITE : เว็บไซต์จำแนกประเภทหิน เพื่อการศึกษาทางธรณีวิทยา</p>', unsafe_allow_html=True)

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
