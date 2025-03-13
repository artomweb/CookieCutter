import json
import cv2
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
from tqdm import tqdm

image_dir = "images/"
json_dir = "masks/"
output_dir = "outputMasks/"
os.makedirs(output_dir, exist_ok=True)

def json_to_mask(image_path, json_path, output_path, visualize=False):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")
    h, w = img.shape[:2]
    
    with open(json_path, "r") as f:
        data = json.load(f)
    
    mask = np.zeros((h, w), dtype=np.uint8)
    
    temp_mask = np.zeros((h, w), dtype=np.uint8)
    cutter_found = False
    for shape in data["shapes"]:
        if shape["label"] == "cutter":
            cutter_found = True
            points = np.array(shape["points"], dtype=np.int32)
            cv2.fillPoly(temp_mask, [points], 1) 
    
    if not cutter_found:
        print(f"Warning: No 'cutter' shapes found in {json_path}")
    
    # Create edge class (Class 1) as a continuous line
    cutter_mask = (temp_mask == 1).astype(np.uint8)
    edge_thickness_factor = 0.04
    kernel_size = max(3, int(w * edge_thickness_factor))
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(cutter_mask, kernel, iterations=1)
    edge = dilated - cutter_mask
    mask[edge == 1] = 1
    
    
    # Save mask with values [0, 255]
    mask_to_save = mask * 255  # Scale to 0-255
    cv2.imwrite(output_path, mask_to_save)
    

    saved_mask = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)
    if saved_mask is None:
        raise ValueError(f"Failed to load saved mask: {output_path}")


    if visualize:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        edge_vis = (mask == 1).astype(np.uint8) * 255
        background_vis = (mask == 0).astype(np.uint8) * 255

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 3, 1)
        plt.title("Input Image")
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.title("Edge Mask (Class 1)")
        plt.imshow(edge_vis, cmap='gray')
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.title("Background Mask (Class 0)")
        plt.imshow(background_vis, cmap='gray')
        plt.axis('off')
        plt.show()


image_files = (
    glob(os.path.join(image_dir, "*.jpg")) +
    glob(os.path.join(image_dir, "*.jpeg")) +
    glob(os.path.join(image_dir, "*.png"))
)

visualized = False
valid_pairs = [(img_path, os.path.join(json_dir, f"{os.path.basename(img_path).split('.')[0]}.json"), 
                os.path.join(output_dir, f"{os.path.basename(img_path).split('.')[0]}_mask.png")) 
               for img_path in image_files 
               if os.path.exists(os.path.join(json_dir, f"{os.path.basename(img_path).split('.')[0]}.json"))]

for img_path, json_path, mask_path in tqdm(valid_pairs, desc="Processing images"):
    if not visualized:
        json_to_mask(img_path, json_path, mask_path, visualize=True)
        visualized = True
    else:
        json_to_mask(img_path, json_path, mask_path, visualize=False)

# Report missing JSON files
missing_json = [img_path for img_path in image_files 
                if os.path.join(json_dir, f"{os.path.basename(img_path).split('.')[0]}.json") 
                not in [jp for _, jp, _ in valid_pairs]]