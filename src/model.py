import torch.nn as nn
from torchvision import models

def build_model(model_name="mobilenet_v3_large"):
    if model_name == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
        model = models.mobilenet_v3_large(weights=weights)
        
        # Freeze backbone initially
        for p in model.features.parameters():
            p.requires_grad = False
            
        # Replace classifier
        in_features = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 2)
        )
    return model
