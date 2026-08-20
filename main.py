import io
import json
import os

# Khắc phục warning thư mục cấu hình của Ultralytics trên môi trường cloud (Railway)
os.environ['YOLO_CONFIG_DIR'] = '/tmp/Ultralytics'

from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from dotenv import load_dotenv

from inference_sdk import InferenceHTTPClient

app = FastAPI(title="Trash2Treasure Vision Hybrid Backend with Roboflow AR")

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

# Khởi tạo Roboflow Client cho AR Model ("waste-detection-vqkjo/3")
roboflow_client = None
if ROBOFLOW_API_KEY:
    try:
        roboflow_client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=ROBOFLOW_API_KEY
        )
        print("✅ Đã kết nối thành công với Roboflow Serverless API!")
    except Exception as e:
        print(f"⚠️ Không thể khởi tạo Roboflow Client: {e}")
else:
    print("⚠️ Cảnh báo: ROBOFLOW_API_KEY chưa được thiết lập trong biến môi trường!")


@app.post("/api/ar-detect")
async def ar_detect_waste(file: UploadFile = File(...)):
    """
    Endpoint AR Scanner: Sử dụng hoàn toàn Roboflow Cloud Model (waste-detection-vqkjo/3)
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File phải là hình ảnh!")
    
    if not roboflow_client:
        return {"has_waste": False, "error": "Roboflow API Key chưa được cấu hình trong Environment Variables của Railway"}

    temp_path = "temp_ar_frame.jpg"
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image.save(temp_path)

        # Gọi mô hình Object Detection trên Roboflow Serverless Cloud API
        response = roboflow_client.infer(temp_path, model_id="waste-detection-vqkjo/3")

        if "predictions" in response and len(response["predictions"]) > 0:
            pred = response["predictions"][0] # Lấy vật thể có độ tự tin cao nhất
            img_w, img_h = image.size
            
            # Tính toán tọa độ hộp bao (bounding box) theo % màn hình
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

            waste_label = pred["class"]
            confidence = round(float(pred["confidence"]), 2)

            # Danh sách ý tưởng DIY trả về cho giao diện
            diy_ideas_list = [
                {
                    "id": "1",
                    "title": f"Làm chậu cây mini từ {waste_label}",
                    "description": f"Rửa sạch, cắt phần thân {waste_label} và đục lỗ thoát nước."
                },
                {
                    "id": "2",
                    "title": f"Làm ống cắm bút sáng tạo",
                    "description": f"Dùng giấy màu trang trí xung quanh {waste_label} để đựng bút."
                },
                {
                    "id": "3",
                    "title": f"Đồ trang trí handmade",
                    "description": f"Kết hợp {waste_label} với các vật liệu khác để tạo điểm nhấn."
                }
            ]

            return {
                "has_waste": True,
                "waste_type": waste_label,
                "category": "Rác tái chế",
                "confidence": confidence,
                "box": box_pct,
                "quick_guide": f"Phân loại và tái chế {waste_label} sáng tạo ♻️",
                "materials": [waste_label, "Dụng cụ cơ bản"],
                "diy_ideas": diy_ideas_list
            }

        return {"has_waste": False}
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
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File phải là hình ảnh!")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        is_kids = children_mode.lower() == "true"
        is_en = lang.lower() == "en"

        # 1. Ưu tiên gọi Gemini Cloud Online
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
                print(f"⚠️ Gemini bận/lỗi ({e}). Chuyển sang Roboflow Fallback...")

        # 2. Fallback qua Roboflow nếu Gemini lỗi
        if roboflow_client:
            temp_path = "temp_analyze.jpg"
            image.save(temp_path)
            try:
                response = roboflow_client.infer(temp_path, model_id="waste-detection-vqkjo/3")
                if "predictions" in response and len(response["predictions"]) > 0:
                    pred = response["predictions"][0]
                    waste_name = pred["class"]
                    return {
                        "has_waste": True,
                        "waste_type": waste_name,
                        "confidence": round(float(pred["confidence"]), 2),
                        "category": "Recyclable",
                        "instructions": ["Làm sạch vật liệu"],
                        "diy_ideas": [{
                            "title": f"Sáng tạo từ {waste_name}",
                            "desc": "Dự án tái chế thông minh",
                            "difficulty": "Dễ",
                            "time": "15 mins",
                            "materials": [waste_name, "Kéo", "Keo dán"],
                            "steps": ["Làm sạch vật liệu", "Cắt dán tạo hình"]
                        }],
                        "engine": "Roboflow Serverless Cloud API"
                    }
            except Exception as e:
                print(f"Lỗi Roboflow analyze: {e}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        return {
            "has_waste": False,
            "message": "Không phát hiện rác thải phù hợp."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {"status": "ok", "message": "Trash2Treasure Server Active with Roboflow AR Engine"}
