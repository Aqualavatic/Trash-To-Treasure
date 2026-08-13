from roboflow import Roboflow

# Tải trực tiếp bộ dữ liệu TACO được format chuẩn YOLOv8
rf = Roboflow(api_key="ANONYMOUS") # Hoặc tạo tài khoản Roboflow miễn phí để lấy API Key
project = rf.workspace("vrb").project("taco-trash-annotations-in-context")
dataset = project.version(1).download("yolov8")

print("✅ Đã tải xong dataset về thư mục:", dataset.location)