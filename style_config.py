import streamlit as st

def apply_custom_style():
    st.markdown("""
        <style>
        /* --- ส่วนที่ 1: จัดการหน้าเว็บทั่วไป --- */
        header {visibility: hidden;}
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        [data-testid="stDecoration"] {display:none;}

        .stApp {
            background-image: linear-gradient(rgba(0,0,0,0.4), rgba(0,0,0,0.4)), 
                              url("https://images.wallpaperscraft.com/image/single/beach_rocks_stones_136868_3840x2400.jpg");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }

        /* --- ส่วนที่ 2: หัวข้อ STONE LEN --- */
        .main-title {
            color: #dcb799 !important;
            font-size: 80px !important;
            font-weight: 900;
            text-shadow: 3px 3px 15px rgba(0,0,0,0.8);
            margin-top: -100px !important; 
            text-align: left;
        }

        /* --- ส่วนที่ 3: กล่องอัปโหลดขาวๆ --- */
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

        /* --- ส่วนที่ 4: ปุ่ม Upload File --- */
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

        /* --- ส่วนที่ 5: แถบรายชื่อด้านล่าง --- */
        .footer-bar {
            position: fixed;
            left: 0; bottom: 0; width: 100%;
            background-color: rgba(45, 62, 51, 0.9);
            color: white; text-align: center;
            padding: 10px; font-size: 14px; z-index: 999;
        }
        </style>
    """, unsafe_allow_html=True)
