import base64
import os
import re
import io
import streamlit as st
from groq import Groq
from gtts import gTTS

# ---------------------------------------------------------
# 1. Cấu hình Trang & CSS Custom cho Giao diện Đẹp Như Vercel
# ---------------------------------------------------------
st.set_page_config(
    page_title="Trash2Treasure AI Engine",
    page_icon="♻️",
    layout="centered"
)

st.markdown("""
<style>
    /* Ẩn bớt các element thừa của Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Font & Container */
    .stApp {
        background-color: #F8FAFC;
    }
    
    /* Banner Vercel Header */
    .vercel-banner {
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
        padding: 12px 20px;
        border-radius: 16px;
        border: 1px solid #A7F3D0;
        margin-bottom: 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.08);
    }
    
    /* Card chứa kết quả */
    .result-card {
        background-color: #FFFFFF;
        padding: 24px;
        border-radius: 20px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.05);
        margin-top: 20px;
    }
    
    /* Custom Nút bấm */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 48px;
        font-weight: 600;
        background-color: #10B981 !important;
        color: white !important;
        border: none !important;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #059669 !important;
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

# Lấy API Key
groq_api_key = st.secrets.get("GROQ_API_KEY") or st.sidebar.text_input("Groq API Key:", type="password")

# ---------------------------------------------------------
# 2. Ngôn ngữ & Từ điển
# ---------------------------------------------------------
lang = st.sidebar.radio("🌐 Ngôn ngữ / Language:", ["🇻🇳 Tiếng Việt", "🇬🇧 English"])
child_mode = st.sidebar.checkbox("👶 Chế độ Trẻ em / Kids Mode", value=True)

# Banner kết nối ngược lại Vercel
st.markdown(f"""
    <div class="vercel-banner">
        <span>👈 <a href="https://trashtotreasure-omega.vercel.app/" target="_self" style="color: #059669; font-weight: bold; text-decoration: none;">
            {"Quay lại Vercel Showcase" if lang == "🇻🇳 Tiếng Việt" else "Back to Vercel Showcase"}
        </a></span>
        <span style="color: #047857; font-size: 0.85em; font-weight: 600;">⚡ Live AI Engine</span>
    </div>
""", unsafe_allow_html=True)

if lang == "🇻🇳 Tiếng Việt":
    title = "♻️ Phân Tích & Tái Chế Rác Thải AI"
    subtitle = "Tải ảnh rác thải lên để nhận hướng dẫn tái chế sáng tạo tức thì!"
    upload_label = "Kéo thả hoặc chọn ảnh rác thải:"
    btn_label = "🚀 Phân Tích Ngay"
    tts_label = "🔊 Nghe Hướng Dẫn Giọng Nói (Text-to-Speech)"
    err_key = "Chưa tìm thấy Groq API Key! Vui lòng kiểm tra cài đặt Secrets."
else:
    title = "♻️ AI Waste Analysis & Recycling"
    subtitle = "Upload a waste photo to get instant creative DIY recycling ideas!"
    upload_label = "Drag and drop or browse waste photo:"
    btn_label = "🚀 Analyze Now"
    tts_label = "🔊 Listen to Voice Guide (Text-to-Speech)"
    err_key = "Groq API Key is missing! Please check Secrets settings."

st.title(title)
st.caption(subtitle)

# ---------------------------------------------------------
# 3. Hàm xử lý Lọc Suy Nghĩ AI & Giọng Nói
# ---------------------------------------------------------
def clean_ai_response(text):
    """Lọc bỏ phần <think>...</think> của AI, chỉ giữ kết quả cuối"""
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return cleaned.strip()

def encode_image(uploaded_file):
    return base64.b64encode(uploaded_file.getvalue()).decode('utf-8')

def generate_audio(text, language_code):
    """Tạo audio giọng nói từ văn bản"""
    # Xóa ký tự Markdown để giọng đọc chuẩn hơn
    clean_text = re.sub(r'[*#_\-`]', '', text)
    tts = gTTS(text=clean_text[:500], lang=language_code) # Đọc 500 ký tự đầu tiên
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    return fp

# ---------------------------------------------------------
# 4. Giao diện Upload & Xử lý AI
# ---------------------------------------------------------
uploaded_file = st.file_uploader(upload_label, type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Photo", use_container_width=True)
    
    if st.button(btn_label):
        if not groq_api_key:
            st.error(err_key)
        else:
            with st.spinner("⚡ AI Processing..."):
                try:
                    client = Groq(api_key=groq_api_key)
                    base64_image = encode_image(uploaded_file)
                    
                    target_lang = "Vietnamese" if lang == "🇻🇳 Tiếng Việt" else "English"
                    safety_prompt = (
                        "KIDS MODE IS ON: Strictly NO sharp knives, hot glue guns, or dangerous tools in instructions." 
                        if child_mode else ""
                    )

                    prompt = f"""
                    You are a professional DIY recycling expert. Analyze this photo and respond ONLY in {target_lang}.
                    DO NOT output any reasoning or internal thoughts. Return directly the final clean Markdown result:

                    ### 🔍 1. Identified Materials
                    - List the detected waste items.

                    ### 💡 2. Top 2 DIY Ideas
                    - **Idea 1:** [Project Name] - [Brief Utility]
                    - **Idea 2:** [Project Name] - [Brief Utility]

                    ### 🛠️ 3. Step-by-Step Instructions
                    - **Selected Project:** [Best Project Name]
                    - **Tools Needed:** [List tools]
                    - **Steps:**
                      1. [Step 1]
                      2. [Step 2]
                      3. [Step 3]
                    - **⚠️ Safety Note:** {safety_prompt}
                    """

                    # Gọi Model Vision mới nhất của Groq
                    chat_completion = client.chat.completions.create(
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                                ],
                            }
                        ],
                        model="qwen/qwen3.6-27b",
                        temperature=0.2,
                    )

                    # Lấy và lọc kết quả
                    raw_result = chat_completion.choices[0].message.content
                    final_result = clean_ai_response(raw_result)

                    # Hiển thị kết quả trong Card sạch đẹp
                    st.success("Done!")
                    st.markdown(f'<div class="result-card">', unsafe_allow_html=True)
                    st.markdown(final_result)
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Tạo Text-To-Speech (Đọc Giọng Nói)
                    st.divider()
                    st.subheader(tts_label)
                    audio_lang = 'vi' if lang == "🇻🇳 Tiếng Việt" else 'en'
                    audio_fp = generate_audio(final_result, audio_lang)
                    st.audio(audio_fp, format='audio/mp3')

                except Exception as e:
                    st.error(f"Error: {e}")
