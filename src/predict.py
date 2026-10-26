import torch
import sys
from PIL import Image
from torchvision import transforms
from model import build_model

def predict_image(image_path, threshold=0.8):
    device = torch.device("cpu")
    
    # Load Model
    model = build_model()
    model.load_state_dict(torch.load("artifacts/best_model.pt", weights_only=True))
    model.eval()
    
    # Preprocess Image
    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        return
        
    img_tensor = tfm(img).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.softmax(output, dim=1)[0][1].item() # Probability of Pneumonia
        
    # Apply Threshold
    label = "PNEUMONIA" if prob >= threshold else "NORMAL"
    
    print("\n" + "="*30)
    print(f"Image: {image_path}")
    print(f"Prediction: {label}")
    print(f"Confidence (Pneumonia): {prob*100:.2f}%")
    print("="*30)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <path_to_image>")
    else:
        predict_image(sys.argv[1])
