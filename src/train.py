import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from dataset import get_dataloaders
from model import build_model
import os

def train():
    device = torch.device("cpu")
    data_dir = "data/chest_xray"
    
    train_loader, val_loader, test_loader, class_weights = get_dataloaders(data_dir)
    model = build_model().to(device)
    
    # Unfreeze the last 2 feature blocks for better medical feature learning
    for p in model.features[-1].parameters():
        p.requires_grad = True
    for p in model.features[-2].parameters():
        p.requires_grad = True
        
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # Lower learning rates for unfrozen backbone layers
    optimizer = optim.AdamW([
        {"params": model.features[-2].parameters(), "lr": 5e-6},
        {"params": model.features[-1].parameters(), "lr": 1e-5},
        {"params": model.classifier.parameters(), "lr": 1e-4},
    ], weight_decay=1e-4)
    
    epochs = 8 # Slightly more epochs
    best_auc = 0.0
    
    print("Starting Fine-Tuning with Oversampling...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        model.eval()
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)[:, 1] 
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.numpy())
                
        val_auc = roc_auc_score(all_labels, all_probs)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | Val AUC: {val_auc:.4f}")
        
        if val_auc > best_auc:
            best_auc = val_auc
            os.makedirs("artifacts", exist_ok=True)
            torch.save(model.state_dict(), "artifacts/best_model.pt")
            print("  -> Model Saved!")
            
    print(f"Training Complete. Best Val AUC: {best_auc:.4f}")

if __name__ == "__main__":
    train()
