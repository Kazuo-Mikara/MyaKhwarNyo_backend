from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PlantAI-Backend")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- Explicit class names (order must match your training) -----
PLANT_CLASSES = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

# ----- Load model -----
try:
    model = YOLO("./models/best.pt")  # or best_float16.tflite
    logger.info("✅ YOLOv8 model loaded successfully")
    # Log the model type for debugging
    try:
        test_result = model.predict("https://ultralytics.com/images/bus.jpg", verbose=False)
        if test_result[0].boxes is not None:
            logger.info("Model type: Detection (bounding boxes available)")
        elif test_result[0].probs is not None:
            logger.info("Model type: Classification (probabilities available)")
    except Exception as e:
        logger.warning(f"Could not determine model type: {e}")
except Exception as e:
    logger.error(f"Model load failed: {e}")
    raise RuntimeError("Model initialization error")

@app.get("/")
def root():
    return {"status": "online", "classes": PLANT_CLASSES}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Validate file type
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(400, "Only JPEG or PNG allowed")

    try:
        # 2. Read and convert image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # 3. Run inference
        results = model.predict(source=image, conf=0.25, verbose=False)
        result = results[0]

        # 4. Handle classification model (probabilities)
        if result.probs is not None:
            top1_idx = result.probs.top1
            confidence = float(result.probs.top1conf)
            predicted_label = PLANT_CLASSES[top1_idx].capitalize()
            confidence_pct = round(confidence * 100, 1)
            logger.info(f"Predicted: {predicted_label} ({confidence_pct}%)")
            return {
                "success": True,
                "commonName": predicted_label,
                "confidence": confidence_pct
            }

        # 5. Handle detection model (bounding boxes)
        if result.boxes is not None and len(result.boxes) > 0:
            best_conf = -1.0
            best_idx = None
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf > best_conf:
                    best_conf = conf
                    best_idx = int(box.cls[0])
            predicted_label = PLANT_CLASSES[best_idx].capitalize()
            confidence_pct = round(best_conf * 100, 1)
            logger.info(f"Predicted: {predicted_label} ({confidence_pct}%)")
            return {
                "success": True,
                "commonName": predicted_label,
                "confidence": confidence_pct
            }

        # 6. No detection / no classification
        return {
            "success": False,
            "message": "No plant detected",
            "commonName": None,
            "confidence": 0.0
        }

    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(500, f"Inference failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)