import streamlit as st

def apply_custom_style():
    st.markdown("""
        <style>
        /* ซ่อนส่วนประกอบของระบบ Streamlit */
        header {visibility: hidden;}
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}

        /* จัดการภาพพื้นหลัง */
        .stApp {
            background-image: linear-gradient(rgba(0,0,0,0.4), rgba(0,0,0,0.4)), 
                              url("https://images.wallpaperscraft.com/image/single/beach_rocks_stones_136868_3840x2400.jpg");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }

        /* ปรับตำแหน่ง STONE LEN (ยิ่งค่าลบเยอะ ยิ่งขึ้นสูง) */
        .main-title {
            color: #dcb799 !important;
            font-size: 100px !important;
            font-weight: 900;
            text-shadow: 3px 3px 15px rgba(0,0,0,0.8);
            margin-top: -60px !important; 
            text-align: left;
        }

        /* ปรับขนาดกล่องสีขาว (ปัจจุบันคือ 350px) */
        [data-testid="stFileUploader"] {
            width: 350px !important; 
            margin: 0 auto !important;
        }
        
        /* ...ส่วนอื่นๆ ถูกต้องดีแล้วครับ... */
        </style>
    """, unsafe_allow_html=True)
/* --- ส่วนที่ 5: แถบรายชื่อด้านล่าง --- */
        .footer-bar {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: rgba(45, 62, 51, 0.95); /* เพิ่มความเข้ม */
            color: white;
            text-align: center;
            padding: 12px 0; /* เพิ่มพื้นที่แนวตั้ง */
            font-size: 14px;
            z-index: 9999; /* บังคับให้ลอยทับทุกองค์ประกอบ */
            border-top: 1px solid rgba(255,255,255,0.1);
        }
