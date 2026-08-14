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

# Cấu hình CORS cho phép mọi nguồn gọi vào (hoặc đổi thành domain Vercel của bạn)
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
yolo_model = None

if os.path.exists(MODEL_PATH):
    try:
        yolo_model = YOLO(MODEL_PATH)
        print("✅ Đã load thành công mô hình best.pt (YOLO)!")
    except Exception as e:
        print(f"⚠️ Chưa thể nạp mô hình best.pt: {e}")
else:
    print(f"⚠️ Không tìm thấy file model tại {MODEL_PATH}")


def run_yolo_inference(image: Image.Image) -> tuple[str, float]:
    """Hàm dự đoán bằng file best.pt"""
    if not yolo_model:
        return "Plastic Bottle", 0.85

    results = yolo_model(image, conf=0.4)
    best_label = None
    max_conf = 0.0

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf > max_conf:
                max_conf = conf
                class_id = int(box.cls[0])
                best_label = yolo_model.names[class_id]

    if best_label:
        return best_label, max_conf

    return "Unknown Waste", 0.0


def generate_local_slm_diy(waste_type: str, is_kids: bool, is_en: bool) -> dict:
    """Sinh ý tưởng DIY an toàn không bị chết khi chạy trên Cloud (Railway)"""
    ollama_url = "http://localhost:11434/api/generate"
    
    # Payload kiểm tra nhanh nếu chạy local có ollama
    payload = {
        "model": "qwen2.5:1.5b", 
        "prompt": f"Generate 1 creative DIY project for '{waste_type}' in RAW JSON format.", 
        "stream": False
    }

    try:
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(ollama_url, data=data, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=1) as response: # Timeout cực nhanh 1s để không bị nghẽn mạng trên Cloud
            res_body = json.loads(response.read().decode('utf-8'))
            raw_res = res_body.get("response", "").strip()
            if "```json" in raw_res: 
                raw_res = raw_res.split("```json")[1].split("```")[0].strip()
            return json.loads(raw_res)
    except Exception:
        # Fallback an toàn 100% trên Cloud khi không tìm thấy Ollama
        return {
            "title": f"Chậu cây mini từ {waste_type}" if not is_en else f"Mini Planter from {waste_type}",
            "desc": "Dự án tái chế sáng tạo và hữu ích tại nhà" if not is_en else "Simple and creative indoor recycling project",
            "difficulty": "Rất Dễ" if is_kids else "Dễ",
            "time": "15 mins",
            "materials": [waste_type, "Kéo / Dao an toàn", "Màu vẽ"] if not is_en else [waste_type, "Scissors", "Colors"],
            "steps": [
                "Làm sạch và lau khô vật liệu." if not is_en else "Clean and dry the material.",
                "Cắt tỉa tạo hình theo sở thích." if not is_en else "Cut and shape as desired.",
                "Trang trí và sử dụng." if not is_en else "Decorate and use."
            ]
        }


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

        # -------------------------------------------------------------
        # BRANCH 1: ONLINE (Gemini Cloud)
        # -------------------------------------------------------------
        if client:
            try:
                print("🌐 [ONLINE] Gọi Cloud Gemini...")
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

                # Sử dụng chuẩn model Gemini phổ biến hiện tại trên Cloud
                response = client.models.generate_content(
                    model='gemini-3.6-flash',
                    contents=[image, prompt],
                    config=types.GenerateContentConfig(response_mime_type="application/json")
                )

                result_json = json.loads(response.text)
                result_json["engine"] = "Gemini Cloud Online"
                return result_json

            except Exception as e:
                print(f"⚠️ Gemini bận/lỗi ({e}). Chuyển sang Local YOLO...")

        # -------------------------------------------------------------
        # BRANCH 2: LOCAL / OFFLINE FALLBACK
        # -------------------------------------------------------------
        print("⚡ [LOCAL] Chạy YOLO & Fallback DIY...")
        detected_waste, confidence = run_yolo_inference(image)
        
        if detected_waste == "Unknown Waste" or confidence == 0:
            return {
                "has_waste": False,
                "message": "Không tìm thấy rác thải trong ảnh." if not is_en else "No waste detected.",
                "engine": "YOLO Edge Engine"
            }

        generated_diy = generate_local_slm_diy(detected_waste, is_kids, is_en)

        return {
            "has_waste": True,
            "waste_type": detected_waste,
            "confidence": round(confidence, 2),
            "category": "Rác tái chế" if not is_en else "Recyclable",
            "instructions": ["Làm sạch vật liệu", "Phân loại theo quy định"],
            "diy_ideas": [generated_diy],
            "engine": "YOLO best.pt & Fallback Engine"
        }

    except Exception as e:
        print(f"❌ Lỗi xử lý chung: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý hệ thống: {str(e)}")


@app.get("/")
def root():
    return {"status": "ok", "message": "Trash2Treasure Server Active"}
