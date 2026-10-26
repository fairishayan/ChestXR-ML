import torch
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from dataset import get_dataloaders
from model import build_model

def evaluate():
    device = torch.device("cpu")
    data_dir = "data/chest_xray"
    
    # Load data
    _, _, test_loader, _ = get_dataloaders(data_dir, batch_size=32)
    
    # Load model
    model = build_model()
    model.load_state_dict(torch.load("artifacts/best_model.pt", weights_only=True))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print("Evaluating on Test Set (this may take a minute)...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            
            # Get predicted class (0=Normal, 1=Pneumonia)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    # Calculate metrics
    print("\n" + "="*40)
    print("TEST SET RESULTS")
    print("="*40)
    print(f"Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=["Normal", "Pneumonia"]))
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print("="*40)

if __name__ == "__main__":
    evaluate()
