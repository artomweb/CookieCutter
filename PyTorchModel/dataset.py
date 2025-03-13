from torch.utils.data import Dataset
from glob import glob
import os
import cv2
import numpy as np
import torch

class EdgeSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_filenames=None, target_size=(512, 512), transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.target_size = target_size
        self.transform = transform

        
        if image_filenames:
            image_files = [os.path.join(image_dir, f) for f in image_filenames if os.path.exists(os.path.join(image_dir, f))]
        else:
            image_extensions = ["*.jpg", "*.jpeg", "*.png"]
            image_files = []
            for ext in image_extensions:
                image_files.extend(glob(os.path.join(image_dir, ext)))
            
        mask_files = glob(os.path.join(mask_dir, "*_mask.png"))
        
        image_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in image_files}
        mask_dict = {os.path.splitext(os.path.basename(f))[0].replace('_mask', ''): f for f in mask_files}
        
        common_keys = set(image_dict.keys()) & set(mask_dict.keys())
        self.pairs = [(image_dict[key], mask_dict[key]) for key in common_keys]
        print(f"Found {len(self.pairs)} valid image-mask pairs")
        
    def __len__(self):
        return len(self.pairs)
        
    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        # Load image and mask in grayscale as NumPy arrays
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None or mask is None:
            raise FileNotFoundError(f"Failed to load image or mask at {img_path} or {mask_path}")

        # Resize using cv2
        image = cv2.resize(image, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_NEAREST)

        # Apply augmentations with albumentations 
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        # Ensure mask is binary
        mask = (mask > 0.5).float()  # Should already be float from ToTensorV2, but binarize again for safety

        return image, mask, img_path, mask_path