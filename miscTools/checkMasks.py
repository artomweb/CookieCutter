# For when mistakes are made and images and masks are lost

import os

def check_images_and_masks(images_dir="images", masks_dir="masks"):
    image_extensions = {'.jpg', '.jpeg', '.png'}
    mask_extension = '.json'
    
    image_files = [f for f in os.listdir(images_dir) 
                  if os.path.splitext(f.lower())[1] in image_extensions]
    
    mask_files = [f for f in os.listdir(masks_dir) 
                 if os.path.splitext(f.lower())[1] == mask_extension]
    
    image_bases = {os.path.splitext(f)[0] for f in image_files}
    mask_bases = {os.path.splitext(f)[0] for f in mask_files}
    
    extra_images = image_bases - mask_bases
    if extra_images:
        print("Images without corresponding masks:")
        for img in extra_images:
            print(f"- {img}")
        
        while True:
            response = input("\nDelete all extra images? (y/n): ").lower().strip()
            if response in ['y', 'n']:
                break
            print("Please enter 'y' or 'n'")
        
        if response == 'y':
            for img in extra_images:
                # Find the actual file with extension
                for ext in image_extensions:
                    img_file = f"{img}{ext}"
                    img_path = os.path.join(images_dir, img_file)
                    if os.path.exists(img_path):
                        try:
                            os.remove(img_path)
                            print(f"Deleted: {img_file}")
                        except Exception as e:
                            print(f"Error deleting {img_file}: {e}")
                        break
            print("Image deletion complete")
        else:
            print("No images deleted")
    else:
        print("All images have corresponding masks")
    
    extra_masks = mask_bases - image_bases
    if extra_masks:
        print("\nMasks without corresponding images:")
        for mask in extra_masks:
            print(f"- {mask}")
        
        while True:
            response = input("\nDelete all extra masks? (y/n): ").lower().strip()
            if response in ['y', 'n']:
                break
            print("Please enter 'y' or 'n'")
        
        if response == 'y':
            for mask in extra_masks:
                mask_file = f"{mask}{mask_extension}"
                mask_path = os.path.join(masks_dir, mask_file)
                try:
                    os.remove(mask_path)
                    print(f"Deleted: {mask_file}")
                except Exception as e:
                    print(f"Error deleting {mask_file}: {e}")
            print("Mask deletion complete")
        else:
            print("No masks deleted")
    else:
        print("\nNo extra masks found")
    
    image_files_after = [f for f in os.listdir(images_dir) 
                        if os.path.splitext(f.lower())[1] in image_extensions]
    mask_files_after = [f for f in os.listdir(masks_dir) 
                       if os.path.splitext(f.lower())[1] == mask_extension]
    image_bases_after = {os.path.splitext(f)[0] for f in image_files_after}
    mask_bases_after = {os.path.splitext(f)[0] for f in mask_files_after}
    
    print(f"\nTotal images: {len(image_bases_after)}")
    print(f"Total masks: {len(mask_bases_after)}")
    print(f"Images without masks: {len(image_bases_after - mask_bases_after)}")
    print(f"Masks without images: {len(mask_bases_after - image_bases_after)}")

if __name__ == "__main__":
    # Verify directories exist
    if not os.path.exists("images"):
        print("Error: 'images' directory not found")
    elif not os.path.exists("masks"):
        print("Error: 'masks' directory not found")
    else:
        check_images_and_masks()