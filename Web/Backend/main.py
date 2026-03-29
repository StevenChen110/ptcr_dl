import io
import base64
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image

app = FastAPI()

# 1. 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. 加载模型
model = YOLO("weights/best.pt") 

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 读取前端传来的图片字节流
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 3. YOLO 推理
    results = model.predict(source=img, conf=0.05)
    
    # 4. 获取带有检测框的结果图
    # results[0].plot() 会返回一个带有检测框和标签的 numpy 数组 (BGR格式)
    res_plotted = results[0].plot()

    # 5. 将结果图编码为 Base64 字符串返回给前端
    _, buffer = cv2.imencode('.jpg', res_plotted)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    # 也可以同时提取检测到的缺陷信息（如数量、类别）
    detections = []
    for box in results[0].boxes:
        detections.append({
            "class": model.names[int(box.cls)],
            "confidence": float(box.conf)
        })

    return {
        "success": True,
        "image": f"data:image/jpeg;base64,{img_base64}",
        "count": len(detections),
        "details": detections
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)