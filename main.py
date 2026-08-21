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

from inference_sdk import InferenceHTTPClient

app = FastAPI(title="UpcycleDIY Hybrid Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

roboflow_client = None
if ROBOFLOW_API_KEY:
    try:
        roboflow_client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=ROBOFLOW_API_KEY
        )
        print("✅ Kết nối Roboflow/YOLO11 thành công!")
    except Exception as e:
        print(f"⚠️ Lỗi khởi tạo Roboflow Client: {e}")


@app.post("/api/ar-detect")
async def ar_detect_waste(file: UploadFile = File(...)):
    """
    AR-SCANNER ENDPOINT (YOLO11 via Roboflow): 
    - Nhận diện vị trí bounding box real-time liên tục với model TredNR/yolo11n_object365.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File phải là hình ảnh!")
    
    temp_path = "temp_ar_frame.jpg"
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image.save(temp_path)

        predictions = []
        if roboflow_client:
            try:
                # Thay đổi model_id tại đây sang mô hình YOLO11 Object365
                response = roboflow_client.infer(temp_path, model_id="coco-dataset-vdnr1/41")
                if "predictions" in response:
                    predictions = response["predictions"]
            except Exception as e:
                print(f"⚠️ Lỗi gọi mô hình AR: {e}")

        img_w, img_h = image.size
        detected_objects = []
        all_labels = []

        for pred in predictions:
            x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
            xmin = max(0, x - w / 2)
            ymin = max(0, y - h / 2)
            xmax = min(img_w, x + w / 2)
            ymax = min(img_h, y + h / 2)

            box_pct = [
                round((ymin / img_h) * 100, 1),
                round((xmin / img_w) * 100, 1),
                round((ymax / img_h) * 100, 1),
                round((xmax / img_w) * 100, 1)
            ]

            raw_label = pred["class"]
            confidence = round(float(pred["confidence"]), 2)
            all_labels.append(raw_label)

            detected_objects.append({
                "waste_type": raw_label,
                "confidence": confidence,
                "box": box_pct
            })

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
    return {"status": "ok", "message": "UpcycleDIY Server Active (YOLO11 AR + Gemini Snap/Upload)"}
