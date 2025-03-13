import albumentations as A
import torch
import torch.nn as nn
from dataset import EdgeSegmentationDataset
from model import UNET
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm
from loss import OutlineConnectivityLoss
import torch.optim as optim
import cv2
from albumentations.pytorch import ToTensorV2
import numpy as np


def train_model(model, train_loader, val_loader, num_epochs, device):
    model = model.to(device)
    total_positives = 0
    total_negatives = 0
    for _, mask, _, _ in train_loader:
        mask_np = mask.numpy().flatten()
        positives = np.sum(mask_np == 1)
        negatives = np.sum(mask_np == 0)
        total_positives += positives
        total_negatives += negatives
    pos_weight_value = total_negatives / total_positives if total_positives > 0 else 1.0
    print(f"Using pos_weight: {pos_weight_value:.2f}")
    pos_weight = torch.tensor([pos_weight_value]).to(device)
    
    criterion_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion_conn = OutlineConnectivityLoss(smooth=1, connectivity_weight=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
        for images, masks, _, _ in train_loop:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            masks = masks.unsqueeze(1)
            loss_bce = criterion_bce(outputs, masks)
            loss_conn = criterion_conn(outputs, masks)
            loss = 0.8 * loss_bce + 0.2 * loss_conn
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks, _, _ in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                masks = masks.unsqueeze(1)
                loss_bce = criterion_bce(outputs, masks)
                loss_conn = criterion_conn(outputs, masks)
                loss = 0.8 * loss_bce + 0.2 * loss_conn
                val_loss += loss.item() * images.size(0)

        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return model

IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512

train_transform = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(
            mean=[0.0],  
            std=[1.0],
            max_pixel_value=255.0,
        ),
    ToTensorV2(),
    ]
)

val_transforms = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0], 
            std=[1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)

image_dir = "../images/"
mask_dir = "../outputMasks/"
batch_size = 4
num_epochs = 80
test_split = 0.2  # 20% for testing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data loading
train_dataset = EdgeSegmentationDataset(image_dir, mask_dir, transform=train_transform)
val_dataset = EdgeSegmentationDataset(image_dir, mask_dir, transform=val_transforms)

# Train-test split on indices
indices = list(range(len(train_dataset)))
train_indices, test_indices = train_test_split(
    indices, test_size=test_split, random_state=42
)

train_subset = Subset(train_dataset, train_indices)
val_subset = Subset(val_dataset, test_indices)

train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

print(f"Training samples: {len(train_subset)}, Test samples: {len(val_subset)}")

# Model
model = UNET().to(device)

# Train and test
model = train_model(model, train_loader, test_loader, num_epochs=num_epochs, device=device)

# Save the model
torch.save(model.state_dict(), "edge_segmentation_model.pth")
print("Model saved as edge_segmentation_model.pth")