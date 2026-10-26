from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import torch
import torch.nn as nn
from PIL import Image
import io
from torchvision import transforms
from src.model import build_model

app = FastAPI(title="ChestXR API")

# Load Model Globally
model = build_model()
model.load_state_dict(torch.load("artifacts/best_model.pt", weights_only=True))
model.eval()

# Image Preprocessing
tfm = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class PredictionResponse(BaseModel):
    label: str
    confidence: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    # Read image
    image_data = await file.read()
    img = Image.open(io.BytesIO(image_data)).convert("RGB")
    
    # Preprocess
    img_tensor = tfm(img).unsqueeze(0)
    
    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.softmax(output, dim=1)[0][1].item()
        
    label = "PNEUMONIA" if prob >= 0.8 else "NORMAL"
    
    return PredictionResponse(label=label, confidence=prob)

@app.get("/")
def read_root():
    return {"status": "ChestXR API is running!"}
