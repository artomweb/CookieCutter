# For each mask in accepted_masks moves the mask and image to new folders
import os
import shutil

mask_dir = "scraping/accepted_masks"
image_dir = "scraping/accepted_images"

dest_image_dir = "images"
dest_mask_dir = "masks"

os.makedirs(dest_image_dir, exist_ok=True)
os.makedirs(dest_mask_dir, exist_ok=True)

image_extensions = ['.jpg', '.jpeg', '.png']

# Iterate through mask files
for mask_file in os.listdir(mask_dir):
    if mask_file.endswith('.json'):
        # Get the base filename without extension
        base_name = os.path.splitext(mask_file)[0]
        
        # Look for matching image file
        for ext in image_extensions:
            image_file = base_name + ext
            source_image_path = os.path.join(image_dir, image_file)
            
            if os.path.exists(source_image_path):
                # Move mask file
                source_mask_path = os.path.join(mask_dir, mask_file)
                dest_mask_path = os.path.join(dest_mask_dir, mask_file)
                shutil.move(source_mask_path, dest_mask_path)
                
                # Move image file
                dest_image_path = os.path.join(dest_image_dir, image_file)
                shutil.move(source_image_path, dest_image_path)
                
                print(f"Moved {mask_file} to {dest_mask_dir}")
                print(f"Moved {image_file} to {dest_image_dir}")
                break  # Stop checking extensions once we find a match
        else:
            print(f"No matching image found for {mask_file}")

print("File moving completed!")