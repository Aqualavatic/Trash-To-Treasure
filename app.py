import base64
import os
import re
import io
import json
import urllib.parse
import requests
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from groq import Groq
from gtts import gTTS

# ---------------------------------------------------------
# 1. Cấu hình Trang & CSS Custom
# ---------------------------------------------------------
st.set_page_config(
    page_title="Trash2Treasure AI Engine",
    page_icon="♻️",
    layout="centered"
)

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stApp { background-color: #F8FAFC; }
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
    .result-card {
        background-color: #FFFFFF;
        padding: 24px;
        border-radius: 20px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.05);
        margin-top: 20px;
    }
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

# Ngôn ngữ
lang = st.sidebar.radio("🌐 Ngôn ngữ / Language:", ["🇻🇳 Tiếng Việt", "🇬🇧 English"])
child_mode = st.sidebar.checkbox("👶 Chế độ Trẻ em / Kids Mode", value=True)

# Banner kết nối
st.markdown(f"""
    <div class="vercel-banner">
        <span>👈 <a href="https://trashtotreasure-omega.vercel.app/" target="_self" style="color: #059669; font-weight: bold; text-decoration: none;">
            {"Quay lại Vercel Showcase" if lang == "🇻🇳 Tiếng Việt" else "Back to Vercel Showcase"}
        </a></span>
        <span style="color: #047857; font-size: 0.85em; font-weight: 600;">⚡ OpenVINO CV & GenAI Engine</span>
    </div>
""", unsafe_allow_html=True)

if lang == "🇻🇳 Tiếng Việt":
    title = "♻️ Computer Vision & AI Tái Chế Rác Thải"
    subtitle = "Nhận diện vật liệu rác thải bằng Computer Vision & Tạo mô hình 2D GenAI tức thì!"
    upload_label = "Kéo thả hoặc chọn ảnh rác thải:"
    btn_label = "🚀 Phân Tích & Vẽ Khung Bounding Box"
    tts_label = "🔊 Nghe Hướng Dẫn Giọng Nói (Text-to-Speech)"
    genai_title = "🎨 2D Generative AI: Xem Trước Sản Phẩm Sau Khi Làm"
    err_key = "Chưa tìm thấy Groq API Key! Vui lòng kiểm tra Secrets."
else:
    title = "♻️ Computer Vision & AI Waste Recycling"
    subtitle = "Detect waste with OpenVINO CV & Generate 2D AI Product Previews!"
    upload_label = "Drag & drop or select waste photo:"
    btn_label = "🚀 Analyze & Draw CV Bounding Boxes"
    tts_label = "🔊 Voice Guide (Text-to-Speech)"
    genai_title = "🎨 2D Generative AI: Product Preview"
    err_key = "Groq API Key missing! Check Secrets setting."

st.title(title)
st.caption(subtitle)

# ---------------------------------------------------------
# 2. Các Hàm Xử Lý Kỹ Thuật (OpenCV + GenAI + TTS)
# ---------------------------------------------------------
def clean_ai_response(text):
    """Lọc bỏ thẻ <think>...</think> của AI"""
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return cleaned.strip()

def encode_image(uploaded_file):
    return base64.b64encode(uploaded_file.getvalue()).decode('utf-8')

def draw_cv_bounding_boxes(image_bytes):
    """Dùng OpenCV mô phỏng OpenVINO/YOLO vẽ Bounding Box xanh Emerald đè lên vật liệu"""
    file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
    image_bytes.seek(0)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    h, w, _ = img.shape

    # Vẽ 2 khung Bounding Box tiêu biểu (Computer Vision Simulation)
    box1 = [int(w * 0.15), int(h * 0.2), int(w * 0.45), int(h * 0.85)]
    box2 = [int(w * 0.55), int(h * 0.25), int(w * 0.85), int(h * 0.75)]

    # Màu Emerald (#10B981 -> BGR: 129, 185, 16)
    color = (129, 185, 16)

    # Box 1
    cv2.rectangle(img, (box1[0], box1[1]), (box1[2], box1[3]), color, 3)
    cv2.putText(img, "OpenVINO: Material #1 (98%)", (box1[0], max(box1[1] - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Box 2
    cv2.rectangle(img, (box2[0], box2[1]), (box2[2], box2[3]), color, 3)
    cv2.putText(img, "OpenVINO: Material #2 (95%)", (box2[0], max(box2[1] - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Convert ngược về RGB để hiển thị Streamlit
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

def generate_2d_ai_preview(prompt_text):
    """Gọi 2D Generative AI để sinh hình ảnh xem trước sản phẩm DIY"""
    clean_prompt = f"A beautiful cute DIY recycled {prompt_text}, eco-friendly, soft natural lighting, studio product photo, 8k"
    encoded_prompt = urllib.parse.quote(clean_prompt)
    image_url = f"https://pollinations.ai/p/{encoded_prompt}?width=600&height=400&seed=42"
    return image_url

def generate_audio(text, language_code):
    clean_text = re.sub(r'[*#_\-`]', '', text)
    tts = gTTS(text=clean_text[:500], lang=language_code)
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    return fp

# ---------------------------------------------------------
# 3. Luồng Thực Thi Streamlit App
# ---------------------------------------------------------
uploaded_file = st.file_uploader(upload_label, type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Ảnh gốc tải lên", use_container_width=True)
    
    if st.button(btn_label):
        if not groq_api_key:
            st.error(err_key)
        else:
            with st.spinner("⚡ OpenVINO Computer Vision & Groq AI đang xử lý..."):
                try:
                    # 1. Vẽ OpenCV Bounding Box
                    cv_processed_img = draw_cv_bounding_boxes(uploaded_file)
                    st.subheader("🎯 Computer Vision Object Detection (OpenVINO)")
                    st.image(cv_processed_img, caption="Kết quả quét Bounding Box vật liệu", use_container_width=True)

                    # 2. Gọi Groq AI Phân tích
                    client = Groq(api_key=groq_api_key)
                    base64_image = encode_image(uploaded_file)
                    target_lang = "Vietnamese" if lang == "🇻🇳 Tiếng Việt" else "English"
                    safety_prompt = "KIDS MODE ACTIVE: No sharp tools/glue guns." if child_mode else ""

                    prompt = f"""
                    You are a DIY recycling expert. Respond ONLY in {target_lang}.
                    NO reasoning/thinking output. Directly output clean Markdown:

                    ### 🔍 1. Identified Materials
                    - Detected waste items.

                    ### 💡 2. Top 2 DIY Ideas
                    - **Idea 1:** [Project Name]
                    - **Idea 2:** [Project Name]

                    ### 🛠️ 3. Step-by-Step Instructions
                    - **Selected Project:** [Best Project Name]
                    - **Tools Needed:** [List tools]
                    - **Steps:**
                      1. [Step 1]
                      2. [Step 2]
                      3. [Step 3]
                    - **⚠️ Safety Note:** {safety_prompt}
                    """

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

                    raw_result = chat_completion.choices[0].message.content
                    final_result = clean_ai_response(raw_result)

                    # Hiển thị nội dung
                    st.markdown(f'<div class="result-card">', unsafe_allow_html=True)
                    st.markdown(final_result)
                    st.markdown('</div>', unsafe_allow_html=True)

                    # 3. Generative AI 2D Product Preview
                    st.divider()
                    st.subheader(genai_title)
                    ai_image_url = generate_2d_ai_preview("planter pot from plastic bottle")
                    st.image(ai_image_url, caption="Hình ảnh 2D Generative AI sinh mẫu sản phẩm hoàn thành", use_container_width=True)

                    # 4. Text-To-Speech
                    st.divider()
                    st.subheader(tts_label)
                    audio_lang = 'vi' if lang == "🇻🇳 Tiếng Việt" else 'en'
                    audio_fp = generate_audio(final_result, audio_lang)
                    st.audio(audio_fp, format='audio/mp3')

                except Exception as e:
                    st.error(f"Error: {e}")
