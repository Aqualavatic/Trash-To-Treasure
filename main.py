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
        print("✅ Kết nối Roboflow thành công!")
    except Exception as e:
        print(f"⚠️ Lỗi khởi tạo Roboflow Client: {e}")


@app.post("/api/ar-detect")
async def ar_detect_waste(file: UploadFile = File(...)):
    """
    AR-SCANNER ENDPOINT:
    - Bước 1: Dùng YOLO / Roboflow (Model Coco/Instance Segmentation) quét nhanh vật thể trong khung hình.
    - Bước 2: Dùng Gemini MỘT LẦN DUY NHẤT để lập ý tưởng DIY, dụng cụ và chia bước chi tiết dựa trên các vật thể vừa quét.
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
                response = roboflow_client.infer(temp_path, model_id="coco-dataset-vdnr1/41")
                if "predictions" in response:
                    predictions = response["predictions"]
            except Exception as e:
                print(f"⚠️ Lỗi Roboflow AR: {e}")

        if len(predictions) > 0:
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

            combined_diy_ideas = []
            if client:
                try:
                    unique_items = ", ".join(set(all_labels))
                    prompt = f"""I detected these multiple items together in a room/frame using computer vision: [{unique_items}]. 
Combine these materials to suggest 3 creative DIY upcycling craft ideas that use these items simultaneously. 
Return strictly a JSON array of objects with keys: 
- 'id' (string)
- 'title' (string)
- 'description' (string)
- 'materials' (array of strings: danh sách dụng cụ/vật liệu cần dùng)
- 'steps' (array of strings: các bước thao tác chi tiết, trong đó mỗi bước nên nhắc rõ tên vật dụng hoặc hành động để hệ thống camera verify)
In Vietnamese."""
                    
                    gemini_res = client.models.generate_content(
                        model='gemini-3.6-flash',
                        contents=prompt,
                        config=types.GenerateContentConfig(response_mime_type="application/json")
                    )
                    combined_diy_ideas = json.loads(gemini_res.text)
                except Exception as e:
                    print(f"⚠️ Lỗi Gemini sinh ý tưởng AR: {e}")

            if not combined_diy_ideas:
                combined_diy_ideas = [
                    {
                        "id": "1",
                        "title": "Hộp đựng bút tái chế đa năng",
                        "description": "Kết hợp các vật liệu vừa quét để làm hộp đựng dụng cụ học tập.",
                        "materials": ["Vật liệu quét", "Kéo", "Keo dán"],
                        "steps": [
                            "Chuẩn bị các vật liệu sạch sẽ trước mặt camera.",
                            "Sử dụng kéo để cắt gọt định hình vật liệu.",
                            "Dùng keo cố định các bộ phận lại với nhau."
                        ]
                    }
                ]

            return {
                "has_waste": True,
                "objects": detected_objects,
                "waste_type": ", ".join(set(all_labels)),
                "confidence": detected_objects[0]["confidence"],
                "diy_ideas": combined_diy_ideas
            }

        return {"has_waste": False, "message": "Không phát hiện vật thể qua AR."}
    except Exception as e:
        return {"has_waste": False, "error": str(e)}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/api/analyze")
async def analyze_waste_image(
    file: UploadFile = File(...),
    children_mode: str = Form("false"),
    lang: str = Form("vi")
):
    """
    UPLOAD ENDPOINT: Chuyên biệt cho Gemini 3.6 Flash phân tích ảnh tĩnh chi tiết.
    """
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
            except Exception as e:
                print(f"⚠️ Lỗi Gemini Upload: {e}")

        return {
            "has_waste": False,
            "message": "Không thể kết nối dịch vụ phân tích hình ảnh."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {"status": "ok", "message": "UpcycleDIY Backend Active (Roboflow YOLO AR + Gemini 3.6 Upload)"}
