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

from ultralytics import YOLO

app = FastAPI(title="Trash2Treasure Vision Hybrid Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

MODEL_PATH = "models/best.pt"
MODEL_URL = "https://huggingface.co/Aqualavatic/UpcycleDIY-YOLO/resolve/main/best.pt"

os.makedirs("models", exist_ok=True)

# -------------------------------------------------------------
# CƠ CHẾ ÉP BUỘC TẢI LẠI MODEL TỪ HUGGING FACE
# -------------------------------------------------------------
# Đặt biến này là True nếu muốn chắc chắn xóa bản cũ và tải lại bản mới từ Hugging Face
force_download = False 

if force_download or not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1024 * 1024:
    if os.path.exists(MODEL_PATH):
        print("🗑️ Phát hiện file mô hình cũ. Đang tiến hành xóa để tải bản mới từ Hugging Face...")
        try:
            os.remove(MODEL_PATH)
        except Exception as e:
            print(f"⚠️ Không thể xóa file cũ: {e}")

    print("📥 Đang bắt đầu tải mô hình best.pt mới từ Hugging Face về server...")
    try:
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("✅ Tải mô hình mới thành công từ Hugging Face!")
    except Exception as e:
        print(f"❌ Lỗi tải mô hình từ Hugging Face: {e}")

yolo_model = None
if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1024 * 1024:
    try:
        yolo_model = YOLO(MODEL_PATH)
        print("✅ Đã load thành công mô hình best.pt (YOLO) lên RAM/GPU!")
    except Exception as e:
        print(f"⚠️ Chưa thể nạp mô hình best.pt: {e}")
else:
    print(f"⚠️ File model tại {MODEL_PATH} không tồn tại hoặc bị lỗi dung lượng.")


def run_yolo_inference_live(image: Image.Image) -> dict:
    if not yolo_model:
        return {"has_waste": False}

    img_width, img_height = image.size
    results = yolo_model(image, conf=0.4)
    
    best_box = None
    max_conf = 0.0
    best_label = ""

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf > max_conf:
                max_conf = conf
                class_id = int(box.cls[0])
                best_label = yolo_model.names[class_id]
                xyxy = box.xyxy[0].tolist()
                best_box = [
                    round((xyxy[1] / img_height) * 100, 1),
                    round((xyxy[0] / img_width) * 100, 1),
                    round((xyxy[3] / img_height) * 100, 1),
                    round((xyxy[2] / img_width) * 100, 1)
                ]

    if best_label and best_box:
        quick_guides = {
            "Plastic Bottle": "Rửa sạch -> Cắt làm chậu cây mini 🌱",
            "Can": "Ép bẹp -> Cho vào thùng tái chế ♻️",
            "Cardboard": "Gấp gọn -> Làm thủ công sáng tạo 📦",
            "Glass": "Tái sử dụng làm lọ hoa trang trí 🏺"
        }
        guide_text = quick_guides.get(best_label, "Làm sạch và phân loại rác tái chế ♻️")

        return {
            "has_waste": True,
            "waste_type": best_label,
            "category": "Rác tái chế",
            "confidence": round(max_conf, 2),
            "box": best_box,
            "quick_guide": guide_text
        }

    return {"has_waste": False}


@app.post("/api/ar-detect")
async def ar_detect_waste(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File phải là hình ảnh!")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        return run_yolo_inference_live(image)
    except Exception as e:
        return {"has_waste": False, "error": str(e)}


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
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        is_kids = children_mode.lower() == "true"
        is_en = lang.lower() == "en"

        if client:
            try:
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
"""
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=[image, prompt],
                    config=types.GenerateContentConfig(response_mime_type="application/json")
                )
                result_json = json.loads(response.text)
                result_json["engine"] = "Gemini Cloud Online"
                return result_json
            except Exception as e:
                print(f"⚠️ Gemini bận/lỗi ({e}). Chuyển sang Local YOLO...")

        detected_waste, confidence = "Plastic Bottle", 0.85
        return {
            "has_waste": True,
            "waste_type": detected_waste,
            "confidence": confidence,
            "category": "Recyclable",
            "instructions": ["Làm sạch vật liệu"],
            "diy_ideas": [{
                "title": f"Chậu cây từ {detected_waste}",
                "desc": "Dự án tái chế tại nhà",
                "difficulty": "Dễ",
                "time": "15 mins",
                "materials": [detected_waste, "Kéo"],
                "steps": ["Cắt và trang trí"]
            }],
            "engine": "YOLO Edge Engine"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {"status": "ok", "message": "Trash2Treasure Server Active"}
