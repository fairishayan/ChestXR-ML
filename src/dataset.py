import os
import torch
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import datasets, transforms

def get_dataloaders(data_dir, batch_size=32):
    train_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomAffine(degrees=7, translate=(0.05, 0.05)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    
    full_train_dataset = datasets.ImageFolder(train_dir, transform=train_tfms)
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    
    val_dataset.dataset.transform = val_tfms
    test_dataset = datasets.ImageFolder(test_dir, transform=val_tfms)
    
    # --- OVERSAMPLING ADDED HERE ---
    # Get labels of the training subset
    train_targets = [full_train_dataset.targets[i] for i in train_dataset.indices]
    class_sample_count = torch.tensor([train_targets.count(0), train_targets.count(1)])
    
    # Weight for each class: inversely proportional to its frequency
    weight = 1. / class_sample_count.float()
    samples_weight = torch.tensor([weight[t] for t in train_targets])
    
    # Sampler that pulls equal amounts of both classes
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    weights = torch.tensor([5216 / (2 * 1341), 5216 / (2 * 3875)], dtype=torch.float)
    
    return train_loader, val_loader, test_loader, weights
