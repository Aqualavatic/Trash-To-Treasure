import streamlit as st
import google.generativeai as genai
from PIL import Image

st.set_page_config(
    page_title="Trash2Treasure AI Engine",
    page_icon="♻️",
    layout="centered"
)

# Thêm Banner điều hướng ngược về Vercel
st.markdown("""
    <div style="background-color: #ECFDF5; padding: 10px 16px; border-radius: 12px; border: 1px solid #A7F3D0; margin-bottom: 20px;">
        <span>👈 <a href="https://trashtotreasure-omega.vercel.app/" target="_self" style="color: #059669; font-weight: bold; text-decoration: none;">Quay lại Trang chủ</a></span>
        <span style="float: right; color: #047857; font-size: 0.85em;">Powered by Intel OpenVINO & Gemini AI</span>
    </div>
""", unsafe_allow_html=True)

# Cấu hình trang Streamlit
st.set_page_config(page_title="Trash-to-Treasure Vision", page_icon="♻️", layout="centered")

st.title("♻️ Trash-to-Treasure Vision")
st.write("Biến rác thải sinh hoạt thành đồ dùng sáng tạo với sự hỗ trợ của AI!")

# Sidebar chứa cấu hình API & Chế độ
st.sidebar.header("⚙️ Cấu hình")
api_key = st.sidebar.text_input("Nhập Gemini API Key:", type="password")
child_mode = st.sidebar.checkbox("👶 Chế độ Trẻ em (An toàn)", value=False)

# Nút tải ảnh
uploaded_file = st.file_uploader("Chụp hoặc tải ảnh rác thải lên:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Ảnh vật liệu đã tải lên", use_container_width=True)
    
    if st.button("🚀 Phân tích & Tạo gợi ý DIY"):
        if not api_key:
            st.error("Vui lòng nhập Gemini API Key ở thanh bên trái!")
        else:
            with st.spinner("AI đang phân tích vật liệu và vạch ra ý tưởng..."):
                try:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    
                    # Prompt tối ưu cho bài toán
                    safety_instruction = "LƯU Ý: Đang bật chế độ trẻ em, KHÔNG gợi ý sử dụng dao nhọn, kéo sắc, keo nến nóng hoặc vật dụng nguy hiểm." if child_mode else ""
                    
                    prompt = f"""
                    Bạn là một chuyên gia tái chế DIY sáng tạo. Hãy phân tích hình ảnh này và trả về kết quả theo định dạng Markdown rõ ràng:
                    
                    ### 1. 🔍 Vật liệu nhận diện được
                    - Liệt kê các loại rác thải/vật liệu nhìn thấy trong ảnh.

                    ### 2. 💡 2 Ý tưởng tái chế sáng tạo nhất
                    - Ý tưởng 1: Tên món đồ + Công dụng ngắn gọn.
                    - Ý tưởng 2: Tên món đồ + Công dụng ngắn gọn.

                    ### 3. 🛠️ Hướng dẫn từng bước làm món đồ tốt nhất
                    - **Dụng cụ cần chuẩn bị:** (kéo, hồ dán, thước...)
                    - **Các bước thực hiện:** (Bước 1, Bước 2, Bước 3...)
                    - **⚠️ Cảnh báo an toàn:** {safety_instruction}
                    """
                    
                    response = model.generate_content([prompt, image])
                    st.success("Hoàn tất!")
                    st.markdown(response.text)
                    
                except Exception as e:
                    st.error(f"Đã xảy ra lỗi: {e}")
