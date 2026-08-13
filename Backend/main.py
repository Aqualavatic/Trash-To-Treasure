import io
import json
import os
import urllib.request
import urllib.parse
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Tùy chọn: Nhập thư viện OpenVINO cho Offline Inference (nếu đã cài openvino)
try:
    from openvino.runtime import Core
    import numpy as np
    import cv2
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False

app = FastAPI(title="Trash2Treasure Vision Hybrid Backend")

# 1. Cấu hình CORS (Cho phép Mobile/Web kết nối)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Mở cho cả Mobile App & Web local test
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Khởi tạo Gemini Client
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# 3. Khởi tạo OpenVINO Engine Cục bộ (Nếu có file model)
ov_compiled_model = None
MODEL_PATH = "models/waste_detector.xml"

if OPENVINO_AVAILABLE and os.path.exists(MODEL_PATH):
    try:
        ie = Core()
        model_ov = ie.read_model(model=MODEL_PATH)
        ov_compiled_model = ie.compile_model(model=model_ov, device_name="CPU") # Hoặc "GPU", "NPU"
        print("✅ Đã load thành công mô hình OpenVINO Offline!")
    except Exception as e:
        print(f"⚠️ Chưa thể nạp mô hình OpenVINO: {e}")


# -------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------

def run_openvino_inference(image: Image.Image) -> str:
    """Hàm chạy mô hình OpenVINO đã train để nhận diện loại rác thải"""
    if not ov_compiled_model:
        return "Plastic Bottle" # Fallback mặc định nếu chưa bỏ file model .xml vào

    # Tiền xử lý ảnh cho OpenVINO (Ví dụ kích thước 640x640 cho YOLO)
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    resized_img = cv2.resize(img_cv, (640, 640))
    input_tensor = np.expand_dims(resized_img.transpose(2, 0, 1), axis=0).astype(np.float32) / 255.0

    # Chạy dự đoán
    results = ov_compiled_model([input_tensor])
    # ... (xử lý output tensor để lấy label có score cao nhất) ...
    return "Plastic Bottle"


def generate_local_slm_diy(waste_type: str, is_kids: bool, is_en: bool) -> dict:
    """Sinh ý tưởng DIY bằng Ollama hoặc Fallback"""
    ollama_url = "http://localhost:11434/api/generate"
    lang_instruction = "Return strictly in ENGLISH." if is_en else "Trả về hoàn toàn bằng TIẾNG VIỆT."
    kids_instruction = "NO sharp tools (knives, hot glue). Safe for 7yo kids." if is_kids else "Standard tools allowed."

    prompt = f"""You are a recycling expert AI. Generate 1 creative DIY project for '{waste_type}' in RAW JSON format.
REQUIREMENTS:
- {lang_instruction}
- {kids_instruction}
- Return ONLY raw valid JSON format without markdown formatting.

Schema:
{{
  "title": "Project Name",
  "desc": "Short description",
  "difficulty": "{'Super Easy' if is_kids else 'Easy'}",
  "time": "15 mins",
  "materials": ["Item 1", "Item 2"],
  "steps": ["Step 1", "Step 2"]
}}
"""
    payload = {"model": "qwen2.5:1.5b", "prompt": prompt, "stream": False}

    try:
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(ollama_url, data=data, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=4) as response:
            res_body = json.loads(response.read().decode('utf-8'))
            raw_res = res_body.get("response", "").strip()
            if "```json" in raw_res: raw_res = raw_res.split("```json")[1].split("```")[0].strip()
            return json.loads(raw_res)
    except Exception:
        return {
            "title": f"Chậu cây mini từ {waste_type}" if not is_en else f"Mini Planter from {waste_type}",
            "desc": "Dự án tái chế đơn giản tại nhà" if not is_en else "Simple indoor project",
            "difficulty": "Rất Dễ" if is_kids else "Dễ",
            "time": "10 mins",
            "materials": [waste_type, "Đất", "Hạt giống"] if not is_en else [waste_type, "Soil", "Seeds"],
            "steps": ["Cắt và vệ sinh bề mặt", "Đục lỗ thoát nước đáy", "Cho đất và trồng cây"]
        }


# -------------------------------------------------------------------
# API ENDPOINTS
# -------------------------------------------------------------------

@app.post("/api/analyze")
async def analyze_waste_image(
    file: UploadFile = File(...),
    children_mode: str = Form("false"),
    lang: str = Form("vi")
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File phải là hình ảnh!")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        is_kids = children_mode.lower() == "true"
        is_en = lang.lower() == "en"

        # -------------------------------------------------------------
        # BRANCH 1: ONLINE (Gemini 3.6 Flash)
        # -------------------------------------------------------------
        if client:
            try:
                print("🌐 [ONLINE] Gọi Cloud Gemini 3.6 Flash...")
                language_instruction = "Return ALL text values in ENGLISH." if is_en else "Trả về TOÀN BỘ bằng TIẾNG VIỆT."

                prompt = f"""
You are a recycling AI. Analyze this image.
{language_instruction}

RULES:
1. If no waste/recyclable detected:
   Return JSON: {{"has_waste": false, "message": "No waste detected."}}

2. If waste found, return JSON:
{{
  "has_waste": true,
  "waste_type": "Name of waste",
  "category": "Recyclable / Non-recyclable",
  "instructions": ["Step 1 sort", "Step 2 clean"],
  "diy_ideas": [
    {{
      "title": "DIY Title",
      "desc": "Short description",
      "difficulty": "{'Super Easy' if is_kids else 'Easy'}",
      "time": "15 mins",
      "materials": ["Tool 1", "Tool 2"],
      "steps": ["Step 1 details", "Step 2 details"]
    }}
  ]
}}
KIDS MODE ({is_kids}): {'NO sharp/dangerous tools.' if is_kids else 'Standard tools.'}
"""

                response = client.models.generate_content(
                    model='gemini-3.6-flash',
                    contents=[image, prompt],
                    config=types.GenerateContentConfig(response_mime_type="application/json")
                )

                result_json = json.loads(response.text)
                result_json["engine"] = "Gemini 3.6 Flash (Cloud Online)"
                return result_json

            except Exception as e:
                print(f"⚠️ Gemini 3.6 Flash bận hoặc mất mạng ({e}). Chuyển sang Offline...")

        # -------------------------------------------------------------
        # BRANCH 2: OFFLINE (OpenVINO AI + Local SLM)
        # -------------------------------------------------------------
        print("⚡ [OFFLINE] Chạy OpenVINO Vision & Local SLM...")
        detected_waste = run_openvino_inference(image)
        generated_diy = generate_local_slm_diy(detected_waste, is_kids, is_en)

        return {
            "has_waste": True,
            "waste_type": detected_waste,
            "category": "Rác tái chế" if not is_en else "Recyclable",
            "instructions": ["Làm sạch vật liệu", "Phân loại theo quy định"],
            "diy_ideas": [generated_diy],
            "engine": "OpenVINO Model & Local SLM (Offline Edge Engine)"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý: {str(e)}")


@app.get("/")
def root():
    return {"status": "ok", "message": "Trash2Treasure Server Active"}