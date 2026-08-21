import io
import json
import os

os.environ['YOLO_CONFIG_DIR'] = '/tmp/Ultralytics'

from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from dotenv import load_dotenv
from ultralytics import YOLO

app = FastAPI(title="UpcycleDIY Hybrid Backend (Local YOLO)")

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

local_yolo_model = None
model_path = "models/best.pt"
if os.path.exists(model_path):
    try:
        local_yolo_model = YOLO(model_path)
        print("✅ Đã load local YOLO model từ models/best.pt thành công!")
    except Exception as e:
        print(f"⚠️ Lỗi khởi tạo local YOLO model: {e}")
else:
    print(f"⚠️ Không tìm thấy file mô hình tại {model_path}!")


@app.post("/api/ar-detect")
async def ar_detect_waste(file: UploadFile = File(...)):
    """
    AR-SCANNER ENDPOINT (Local Ultralytics YOLO): 
    - Nhận diện vị trí bounding box real-time bằng mô hình chạy local.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File phải là hình ảnh!")
    
    temp_path = "temp_ar_frame.jpg"
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image.save(temp_path)

        detected_objects = []
        all_labels = []

        if local_yolo_model:
            try:
                results = local_yolo_model(temp_path, verbose=False)
                img_w, img_h = image.size

                for r in results:
                    for box in r.boxes:
                        coords = box.xyxy[0].tolist()
                        xmin, ymin, xmax, ymax = coords
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        raw_label = local_yolo_model.names[cls_id]

                        box_pct = [
                            round((ymin / img_h) * 100, 1),
                            round((xmin / img_w) * 100, 1),
                            round((ymax / img_h) * 100, 1),
                            round((xmax / img_w) * 100, 1)
                        ]

                        detected_objects.append({
                            "waste_type": raw_label,
                            "confidence": round(conf, 2),
                            "box": box_pct
                        })
                        all_labels.append(raw_label)
            except Exception as e:
                print(f"⚠️ Lỗi chạy local YOLO inference: {e}")

        if len(detected_objects) > 0:
            return {
                "has_waste": True,
                "objects": detected_objects,
                "waste_type": ", ".join(set(all_labels)),
                "confidence": detected_objects[0]["confidence"]
            }

        return {"has_waste": False, "message": "Không phát hiện vật thể qua AR."}
    except Exception as e:
        return {"has_waste": False, "error": str(e)}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/api/generate-diy-options")
async def generate_diy_options(
    file: UploadFile = File(...),
    items: str = Form(...),
    lang: str = Form("vi")
):
    """
    ENDPOINT SNAP ẢNH GỌI GEMINI:
    - Nhận ảnh và danh sách vật thể do YOLO quét ổn định.
    - Trả về danh sách 3 ý tưởng DIY kèm theo dụng cụ và các bước thực hiện dựa theo ngôn ngữ.
    """
    if not client:
        raise HTTPException(status_code=500, detail="Gemini Client chưa được khởi tạo!")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        is_en = lang.lower() == "en"
        language_instruction = "Return ALL text values in ENGLISH." if is_en else "Trả về TOÀN BỘ bằng TIẾNG VIỆT."

        prompt = f"""I detected these stable items in the camera frame: [{items}]. 
Combine these materials together to suggest 3 creative DIY upcycling craft ideas that use these items simultaneously. 
{language_instruction}
Return strictly a JSON array of objects with keys: 
- 'id' (string)
- 'title' (string)
- 'description' (string)
- 'materials' (array of strings)
- 'steps' (array of strings)"""

        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=[image, prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        
        diy_ideas = json.loads(response.text)
        return {"success": True, "diy_ideas": diy_ideas}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze")
async def analyze_waste_image(
    file: UploadFile = File(...),
    children_mode: str = Form("false"),
    lang: str = Form("vi")
):
    """
    UPLOAD ENDPOINT: Gemini 3.6 Flash phân tích chuyên sâu ảnh tải lên.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File phải là hình ảnh!")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        is_kids = children_mode.lower() == "true"
        is_en = lang.lower() == "en"

        if client:
            language_instruction = "Return ALL text values in ENGLISH." if is_en else "Trả về TOÀN BỘ bằng TIẾNG VIỆT."
            prompt = f"""
You are an advanced UpcycleDIY AI. Analyze this image.
{language_instruction}
RULES:
1. If no waste detected: Return JSON: {{"has_waste": false, "message": "No waste detected."}}
2. If waste found, return JSON:
{{
  "has_waste": true,
  "waste_type": "Name of waste",
  "category": "Recyclable / Non-recyclable",
  "instructions": ["Step 1", "Step 2"],
  "diy_ideas": [
    {{
      "title": "DIY Title",
      "desc": "Short description",
      "difficulty": "{'Super Easy' if is_kids else 'Easy'}",
      "time": "15 mins",
      "materials": ["Tool 1"],
      "steps": ["Steps details"]
    }}
  ]
}}
"""
            response = client.models.generate_content(
                model='gemini-3.6-flash',
                contents=[image, prompt],
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            result_json = json.loads(response.text)
            result_json["engine"] = "Gemini 3.6 Flash Cloud"
            return result_json

        return {"has_waste": False, "message": "Không thể kết nối AI."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {"status": "ok", "message": "UpcycleDIY Server Active (Local YOLO AR + Gemini Snap/Upload)"}
