import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from dataset import get_dataloaders
from model import build_model

def find_best_threshold():
    device = torch.device("cpu")
    _, _, test_loader, _ = get_dataloaders("data/chest_xray", batch_size=32)
    
    model = build_model()
    model.load_state_dict(torch.load("artifacts/best_model.pt", weights_only=True))
    model.eval()
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1] # Prob of Pneumonia
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Test different thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    print(f"{'Threshold':<10} | {'Accuracy':<10} | {'F1-Score':<10}")
    print("-" * 35)
    
    for t in thresholds:
        preds = (all_probs >= t).astype(int)
        acc = accuracy_score(all_labels, preds)
        f1 = f1_score(all_labels, preds)
        print(f"{t:<10} | {acc:.4f}     | {f1:.4f}")

if __name__ == "__main__":
    find_best_threshold()
