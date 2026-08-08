import base64
import os
import streamlit as st
from groq import Groq
from PIL import Image

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Trash2Treasure AI Engine",
    page_icon="♻️",
    layout="centered"
)

# Banner chuyển hướng ngược về Vercel Showcase
st.markdown("""
    <div style="background-color: #ECFDF5; padding: 12px 18px; border-radius: 12px; border: 1px solid #A7F3D0; margin-bottom: 24px;">
        <span>👈 <a href="https://trashtotreasure-omega.vercel.app/" target="_self" style="color: #059669; font-weight: bold; text-decoration: none;">Quay lại Trang chủ</a></span>
        <span style="float: right; color: #047857; font-size: 0.85em; font-weight: 500;">Powered by Groq LPU & Intel OpenVINO</span>
    </div>
""", unsafe_allow_html=True)

st.title("♻️ Trash2Treasure - Live AI Engine")
st.write("Tải ảnh rác thải sinh hoạt lên để AI phân tích và đề xuất ý tưởng tái chế DIY tức thì!")

# Lấy API Key từ Secrets hoặc Sidebar Input
groq_api_key = st.secrets.get("GROQ_API_KEY") or st.sidebar.text_input("Nhập Groq API Key:", type="password")
child_mode = st.sidebar.checkbox("👶 Chế độ Trẻ em (An toàn)", value=True)

# Khung Upload ảnh
uploaded_file = st.file_uploader("Chọn hoặc chụp ảnh rác thải:", type=["jpg", "jpeg", "png"])

# Hàm chuyển ảnh sang Base64 để gửi cho Groq Vision
def encode_image(uploaded_file):
    return base64.b64encode(uploaded_file.getvalue()).decode('utf-8')

if uploaded_file is not None:
    # Hiển thị ảnh vừa upload
    st.image(uploaded_file, caption="Ảnh vật liệu đã tải lên", use_container_width=True)
    
    if st.button("🚀 Phân tích & Tạo gợi ý DIY với Groq Vision", type="primary"):
        if not groq_api_key:
            st.error("Khuyết API Key! Vui lòng cấu hình GROQ_API_KEY trong Secrets hoặc nhập ở Sidebar.")
        else:
            with st.spinner("⚡ Groq LPU đang phân tích hình ảnh siêu tốc..."):
                try:
                    # Khởi tạo Client Groq
                    client = Groq(api_key=groq_api_key)
                    base64_image = encode_image(uploaded_file)
                    
                    # Chỉ dẫn an toàn nếu bật Chế độ Trẻ em
                    safety_instruction = (
                        "LƯU Ý ĐẶC BIỆT: Chế độ trẻ em đang BẬT. Tuyệt đối KHÔNG hướng dẫn sử dụng "
                        "dao nhọn, kéo sắc, keo nến nóng hoặc bất kỳ dụng cụ nguy hiểm nào."
                        if child_mode else ""
                    )

                    prompt = f"""
                    Bạn là chuyên gia tái chế DIY sáng tạo. Hãy phân tích hình ảnh này và trả về kết quả bằng tiếng Việt theo định dạng Markdown rõ ràng:

                    ### 🔍 1. Vật liệu nhận diện được
                    - Liệt kê các loại rác thải/vật liệu cụ thể nhìn thấy trong ảnh.

                    ### 💡 2. 2 Ý tưởng tái chế sáng tạo nhất
                    - **Ý tưởng 1:** [Tên món đồ] - [Công dụng ngắn gọn]
                    - **Ý tưởng 2:** [Tên món đồ] - [Công dụng ngắn gọn]

                    ### 🛠️ 3. Hướng dẫn từng bước làm món đồ tốt nhất
                    - **Món đồ thực hiện:** [Tên món đồ được chọn]
                    - **Dụng cụ cần chuẩn bị:** [Kéo, keo dán, thước, ...]
                    - **Các bước thực hiện:**
                      1. [Bước 1...]
                      2. [Bước 2...]
                      3. [Bước 3...]
                    - **⚠️ Cảnh báo an toàn:** {safety_instruction}
                    """

                    # Gọi mô hình Groq Llama 3.2 Vision
                    chat_completion = client.chat.completions.create(
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{base64_image}"
                                        },
                                    },
                                ],
                            }
                        ],
                        model="llama-3.3-70b-versatile",
                        temperature=0.2,
                    )

                    # Hiển thị kết quả
                    st.success("Phân tích hoàn tất!")
                    st.markdown(chat_completion.choices[0].message.content)

                except Exception as e:
                    st.error(f"Đã xảy ra lỗi khi gọi Groq API: {e}")
