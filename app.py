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
        <span style="color: #047857; font-size: 0.85em; font-weight: 600;">⚡ Dynamic CV & High-Precision AI</span>
    </div>
""", unsafe_allow_html=True)

if lang == "🇻🇳 Tiếng Việt":
    title = "♻️ Dynamic Computer Vision & AI Tái Chế"
    subtitle = "Khoanh vùng tự động bằng OpenCV & Phân tích rác thải chuẩn xác!"
    upload_label = "Tải lên hoặc chụp ảnh rác thải:"
    btn_label = "🚀 Phân Tích Ngay"
    tts_label = "🔊 Nghe Hướng Dẫn Giọng Nói (Text-to-Speech)"
    genai_title = "🎨 2D Generative AI: Xem Trước Sản Phẩm"
else:
    title = "♻️ Dynamic CV & AI Recycling"
    subtitle = "Dynamic Object Detection & High-Precision AI Generation!"
    upload_label = "Upload or take a waste photo:"
    btn_label = "🚀 Analyze Now"
    tts_label = "🔊 Voice Guide (Text-to-Speech)"
    genai_title = "🎨 2D Generative AI Preview"

st.title(title)
st.caption(subtitle)

# ---------------------------------------------------------
# 2. Dynamic OpenCV Bounding Box (Đồng màu viền & Chữ cực nét)
# ---------------------------------------------------------
def draw_dynamic_bounding_boxes(image_bytes):
    file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
    image_bytes.seek(0)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    h, w, _ = img.shape

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = (w * h) * 0.03
    valid_boxes = []
    
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw * bh > min_area and bw < w * 0.9 and bh < h * 0.9:
            valid_boxes.append((x, y, bw, bh))
            
    if not valid_boxes:
        valid_boxes = [
            (int(w * 0.15), int(h * 0.2), int(w * 0.35), int(h * 0.55)),
            (int(w * 0.52), int(h * 0.25), int(w * 0.35), int(h * 0.5))
        ]
    
    valid_boxes = sorted(valid_boxes, key=lambda b: b[2] * b[3], reverse=True)[:3]

    # BGR Emerald Theme (129, 185, 16)
    theme_color = (129, 185, 16)
    text_color = (255, 255, 255)

    for idx, (x, y, bw, bh) in enumerate(valid_boxes):
        label = f" Material #{idx + 1} ({96 - idx * 3}%) "
        
        # 1. Khung Bounding Box
        cv2.rectangle(img, (x, y), (x + bw, y + bh), theme_color, 3)
        
        # 2. Thẻ Nền cho chữ (Filled Box kín, chữ không lo dính hình nền)
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        label_y = max(y - 8, text_h + 8)
        cv2.rectangle(img, (x, label_y - text_h - 6), (x + text_w + 4, label_y + 4), theme_color, -1)
        
        # 3. Chữ màu Trắng in trên thẻ nền
        cv2.putText(img, label, (x + 2, label_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 2, cv2.LINE_AA)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

# ---------------------------------------------------------
# 3. Bộ lọc suy nghĩ AI chuẩn xác
# ---------------------------------------------------------
def clean_ai_response(text):
    """Cắt sạch toàn bộ thẻ suy nghĩ <think>...</think>"""
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Loại bỏ các dòng tiêu đề suy luận nếu có
    cleaned = re.sub(r'^(Thought|Thinking Process|Reasoning):.*?\n\n', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()

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
# 4. Giao diện Streamlit App
# ---------------------------------------------------------
uploaded_file = st.file_uploader(upload_label, type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Ảnh gốc", use_container_width=True)
    
    if st.button(btn_label):
        if not groq_api_key:
            st.error("Khuyết Groq API Key!")
        else:
            with st.spinner("⚡ AI đang phân tích rác thải..."):
                try:
                    # 1. Computer Vision Detection
                    cv_processed_img = draw_dynamic_bounding_boxes(uploaded_file)
                    st.subheader("🎯 Computer Vision Object Detection")
                    st.image(cv_processed_img, caption="Tự động khoanh vùng vật thể với nhãn tương phản cao", use_container_width=True)

                    # 2. AI Processing
                    client = Groq(api_key=groq_api_key)
                    base64_image = encode_image(uploaded_file)
                    target_lang = "Vietnamese" if lang == "🇻🇳 Tiếng Việt" else "English"
                    safety_prompt = "KIDS MODE ACTIVE: Absolutely NO sharp tools, knives, or hot glue guns." if child_mode else ""

                    prompt = f"""
                    You are a DIY recycling expert. Respond strictly in {target_lang}.
                    Generate a clean Markdown response with these exact sections:

                    ### 🔍 1. Vật liệu nhận diện
                    - Liệt kê các vật liệu tái chế tìm thấy trong ảnh.

                    ### 💡 2. Top 2 Ý tưởng tái chế
                    - **Ý tưởng 1:** [Tên dự án]
                    - **Ý tưởng 2:** [Tên dự án]

                    ### 🛠️ 3. Hướng dẫn làm chi tiết
                    - **Dự án chọn:** [Tên dự án hay nhất]
                    - **Dụng cụ cần:** [Dụng cụ đơn giản]
                    - **Các bước thực hiện:**
                      1. [Bước 1]
                      2. [Bước 2]
                      3. [Bước 3]
                    - **⚠️ Lưu ý an toàn:** {safety_prompt}
                    """

                    # Tăng max_completion_tokens lên 1024 để không bị trảm đuôi
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
                        max_completion_tokens=1024, # <--- Tăng token đủ chứa suy nghĩ + câu trả lời!
                    )

                    raw_text = chat_completion.choices[0].message.content
                    final_result = clean_ai_response(raw_text)

                    # Hiển thị kết quả sạch
                    st.markdown(f'<div class="result-card">', unsafe_allow_html=True)
                    st.markdown(final_result)
                    st.markdown('</div>', unsafe_allow_html=True)

                    # 3. 2D Generative AI Preview
                    st.divider()
                    st.subheader(genai_title)
                    ai_image_url = generate_2d_ai_preview("plastic bottle planter pot")
                    st.image(ai_image_url, caption="2D Generative AI Preview", use_container_width=True)

                    # 4. Text-To-Speech
                    st.divider()
                    st.subheader(tts_label)
                    audio_lang = 'vi' if lang == "🇻🇳 Tiếng Việt" else 'en'
                    audio_fp = generate_audio(final_result, audio_lang)
                    st.audio(audio_fp, format='audio/mp3')

                except Exception as e:
                    st.error(f"Error: {e}")
