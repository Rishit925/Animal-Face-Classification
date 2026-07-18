from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import tempfile
import os

from predict import predict_image

app = FastAPI(
    title="Animal Face Classifier API",
    description="API for classifying animal face images",
    version="1.0"
)

@app.get("/")
def home():
    return {
        "message": "Animal Face Classifier API is running!"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:

        shutil.copyfileobj(file.file, temp_file)

        temp_path = temp_file.name

    label, confidence = predict_image(temp_path)

    os.remove(temp_path)

    return JSONResponse(
        content={
            "prediction": label,
            "confidence": round(float(confidence), 4)
        }
    )