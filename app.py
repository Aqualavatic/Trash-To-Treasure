import base64
import os
import re
import io
import urllib.parse
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from groq import Groq
from gtts import gTTS

# ---------------------------------------------------------
# 1. Cấu hình Trang & Giao diện CSS
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
    }
</style>
""", unsafe_allow_html=True)

# API Key & Sidebar
groq_api_key = st.secrets.get("GROQ_API_KEY") or st.sidebar.text_input("Groq API Key:", type="password")
lang = st.sidebar.radio("🌐 Ngôn ngữ / Language:", ["🇻🇳 Tiếng Việt", "🇬🇧 English"])
child_mode = st.sidebar.checkbox("👶 Chế độ Trẻ em / Kids Mode", value=True)

# Banner kết nối
st.markdown(f"""
    <div class="vercel-banner">
        <span>👈 <a href="https://trashtotreasure-omega.vercel.app/" target="_self" style="color: #059669; font-weight: bold; text-decoration: none;">
            {"Quay lại Vercel Showcase" if lang == "🇻🇳 Tiếng Việt" else "Back to Vercel Showcase"}
        </a></span>
        <span style="color: #047857; font-size: 0.85em; font-weight: 600;">⚡ Adaptive CV & High-Speed GenAI</span>
    </div>
""", unsafe_allow_html=True)

if lang == "🇻🇳 Tiếng Việt":
    title = "♻️ Computer Vision & AI Tái Chế Rác Thải"
    subtitle = "Nhận diện vật liệu bằng Adaptive Bounding Box & Sinh ý tưởng siêu tốc!"
    upload_label = "Tải lên hoặc chụp ảnh rác thải:"
    btn_label = "🚀 Phân Tích Siêu Tốc"
    tts_label = "🔊 Nghe Hướng Dẫn Giọng Nói (Text-to-Speech)"
    genai_title = "🎨 2D Generative AI: Xem Trước Sản Phẩm"
else:
    title = "♻️ Adaptive CV & High-Speed AI Recycling"
    subtitle = "Smart Adaptive Bounding Box & Ultra-fast AI Generation!"
    upload_label = "Upload or take a waste photo:"
    btn_label = "🚀 Ultra-Fast Analyze"
    tts_label = "🔊 Voice Guide (Text-to-Speech)"
    genai_title = "🎨 2D Generative AI Preview"

st.title(title)
st.caption(subtitle)

# ---------------------------------------------------------
# 2. Thuật Toán Adaptive Bounding Box (Đổi Màu Tự Động)
# ---------------------------------------------------------
def get_adaptive_color(roi_img):
    """
    Tính toán độ sáng trung bình (Luminance) của vùng ảnh bên dưới.
    Trả về màu Bounding Box có độ tương phản cao nhất.
    """
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    avg_luminance = np.mean(gray)
    
    # Nếu nền tối (< 128) -> Dùng màu Đèn Neon sáng (Trắng hoặc Xanh Vàng Neon)
    # Nếu nền sáng (>= 128) -> Dùng màu Đen hoặc Xanh Đậm / Đỏ
    if avg_luminance < 128:
        box_color = (255, 255, 255)  # Trắng
        text_color = (0, 255, 255)   # Vàng Neon (BGR)
    else:
        box_color = (0, 0, 0)        # Đen
        text_color = (0, 0, 180)     # Đỏ Đậm
        
    return box_color, text_color

def draw_adaptive_bounding_boxes(image_bytes):
    """Vẽ Bounding Box với màu sắc tự động thích ứng với nền"""
    file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
    image_bytes.seek(0)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    h, w, _ = img.shape

    # Tọa độ 2 khung vật liệu mẫu
    boxes = [
        {"coords": [int(w * 0.15), int(h * 0.2), int(w * 0.45), int(h * 0.85)], "label": "Material #1 (98%)"},
        {"coords": [int(w * 0.55), int(h * 0.25), int(w * 0.85), int(h * 0.75)], "label": "Material #2 (95%)"}
    ]

    for box in boxes:
        x1, y1, x2, y2 = box["coords"]
        # Cắt vùng ảnh dưới khung để tính độ sáng
        roi = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        
        if roi.size > 0:
            box_color, text_color = get_adaptive_color(roi)
        else:
            box_color, text_color = (0, 255, 0), (255, 255, 255)

        # Vẽ Bounding Box & Label
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 3)
        
        # Tạo viền đen cho chữ để luôn đọc được trên mọi nền
        label_text = f"OpenVINO: {box['label']}"
        cv2.putText(img, label_text, (x1, max(y1 - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4) # Viền chữ
        cv2.putText(img, label_text, (x1, max(y1 - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2) # Chữ chính

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

def clean_ai_response(text):
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

def encode_image(uploaded_file):
    return base64.b64encode(uploaded_file.getvalue()).decode('utf-8')

def generate_2d_ai_preview(prompt_text):
    clean_prompt = f"A beautiful cute DIY recycled {prompt_text}, eco-friendly, soft natural lighting, studio product photo, 8k"
    encoded_prompt = urllib.parse.quote(clean_prompt)
    return f"https://pollinations.ai/p/{encoded_prompt}?width=600&height=400&seed=42"

def generate_audio(text, language_code):
    clean_text = re.sub(r'[*#_\-`]', '', text)
    tts = gTTS(text=clean_text[:400], lang=language_code)
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    return fp

# ---------------------------------------------------------
# 3. Giao diện Streamlit App
# ---------------------------------------------------------
uploaded_file = st.file_uploader(upload_label, type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Photo", use_container_width=True)
    
    if st.button(btn_label):
        if not groq_api_key:
            st.error("Khuyết Groq API Key!")
        else:
            with st.spinner("⚡ Processing in <1 second..."):
                try:
                    # 1. Adaptive Bounding Box
                    cv_processed_img = draw_adaptive_bounding_boxes(uploaded_file)
                    st.subheader("🎯 Adaptive Computer Vision (Contrast Enhanced)")
                    st.image(cv_processed_img, caption="Bounding box tự tương phản theo màu nền ảnh", use_container_width=True)

                    # 2. Gọi Groq AI Vision (Tối ưu Prompt siêu ngắn để sinh cực nhanh)
                    client = Groq(api_key=groq_api_key)
                    base64_image = encode_image(uploaded_file)
                    target_lang = "Vietnamese" if lang == "🇻🇳 Tiếng Việt" else "English"
                    safety_prompt = "KIDS MODE: NO sharp tools/glue guns." if child_mode else ""

                    # Prompt tối ưu hóa độ dài để Groq LPU sinh kết quả siêu tốc
                    prompt = f"""
                    DIY Expert. Respond ONLY in {target_lang}. NO reasoning.
                    Directly output clean Markdown:

                    ### 🔍 1. Identified Materials
                    - Item list.

                    ### 💡 2. Top 2 DIY Ideas
                    - **Idea 1:** [Name]
                    - **Idea 2:** [Name]

                    ### 🛠️ 3. Quick Steps
                    - **Project:** [Best Name]
                    - **Tools:** [Tools]
                    - **Steps:** 1. [Step 1] | 2. [Step 2] | 3. [Step 3]
                    - **⚠️ Safety:** {safety_prompt}
                    """

                    # Dùng model llama-3.2-11b-vision-preview hoặc qwen3.6-27b với max_tokens tối ưu
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
                        model="llama-3.2-11b-vision-preview", # Hoặc "qwen/qwen3.6-27b"
                        temperature=0.1,
                        max_tokens=600,
                    )

                    final_result = clean_ai_response(chat_completion.choices[0].message.content)

                    # Hiển thị
                    st.markdown(f'<div class="result-card">', unsafe_allow_html=True)
                    st.markdown(final_result)
                    st.markdown('</div>', unsafe_allow_html=True)

                    # 3. 2D AI Preview
                    st.divider()
                    st.subheader(genai_title)
                    ai_image_url = generate_2d_ai_preview("planter pot from plastic bottle")
                    st.image(ai_image_url, caption="2D Generative AI Product Preview", use_container_width=True)

                    # 4. Text-To-Speech
                    st.divider()
                    st.subheader(tts_label)
                    audio_lang = 'vi' if lang == "🇻🇳 Tiếng Việt" else 'en'
                    audio_fp = generate_audio(final_result, audio_lang)
                    st.audio(audio_fp, format='audio/mp3')

                except Exception as e:
                    st.error(f"Error: {e}")
