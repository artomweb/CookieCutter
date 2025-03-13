# For filtering images in downloaded_images to accepted and rejected
import os
import tkinter as tk
from PIL import Image, ImageTk
import shutil

SOURCE_DIR = "downloaded_images" 
ACCEPTED_DIR = "accepted_images"
REJECTED_DIR = "rejected_images"

# Create output directories if they don't exist
for directory in [ACCEPTED_DIR, REJECTED_DIR]:
    os.makedirs(directory, exist_ok=True)

class ImageFilterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Filter Tool")

        self.canvas = tk.Canvas(root, width=800, height=600)
        self.canvas.pack()

        tk.Label(root, text="A: Accept | R: Reject | Q: Quit").pack()

        # Bind keyboard shortcuts
        self.root.bind("a", self.accept_image)
        self.root.bind("r", self.reject_image)
        self.root.bind("q", self.quit_app)

        # Load images from the downloaded_images folder
        self.image_files = self.load_local_images()
        if not self.image_files:
            self.canvas.create_text(400, 300, text="No images found in downloaded_images folder!", font=("Arial", 20))
            return
        
        self.current_index = 0
        self.load_image()

    def load_local_images(self):
        valid_extensions = (".jpg", ".jpeg", ".png", ".webp")
        image_files = [
            os.path.join(SOURCE_DIR, f) for f in os.listdir(SOURCE_DIR)
            if f.lower().endswith(valid_extensions) and os.path.isfile(os.path.join(SOURCE_DIR, f))
        ]
        print(f"Found {len(image_files)} images in {SOURCE_DIR}")
        return image_files

    def load_image(self):
        if not self.image_files or self.current_index >= len(self.image_files):
            self.canvas.create_text(400, 300, text="No more images to filter!", font=("Arial", 20))
            return

        filepath = self.image_files[self.current_index]
        try:
            self.current_image = Image.open(filepath)
        except Exception as e:
            print(f"Error opening {filepath}: {e}")
            self.current_index += 1
            self.load_image()
            return

        # Resize image to fit canvas while maintaining aspect ratio
        img_width, img_height = self.current_image.size
        canvas_width, canvas_height = 800, 600
        scale = min(canvas_width / img_width, canvas_height / img_height)
        new_size = (int(img_width * scale), int(img_height * scale))
        resized_image = self.current_image.resize(new_size, Image.Resampling.LANCZOS)

        self.photo = ImageTk.PhotoImage(resized_image)
        self.canvas.delete("all")
        self.canvas.create_image(400, 300, image=self.photo)
        self.canvas.create_text(
            400, 20, 
            text=f"Image {self.current_index + 1}/{len(self.image_files)}: {os.path.basename(filepath)}", 
            font=("Arial", 12)
        )

    def accept_image(self, event):
        self.move_image(ACCEPTED_DIR)

    def reject_image(self, event):
        self.move_image(REJECTED_DIR)

    def move_image(self, target_dir):
        if self.current_index < len(self.image_files):
            current_file = self.image_files[self.current_index]
            try:
                shutil.move(current_file, os.path.join(target_dir, os.path.basename(current_file)))
                print(f"Moved {os.path.basename(current_file)} to {target_dir}")
            except Exception as e:
                print(f"Error moving {current_file}: {e}")
            self.current_index += 1
            self.load_image()

    def quit_app(self, event):
        self.root.quit()
        print("Application closed.")

if __name__ == "__main__":
    # Check if source directory exists
    if not os.path.exists(SOURCE_DIR) or not os.listdir(SOURCE_DIR):
        print(f"Error: {SOURCE_DIR} does not exist or is empty. Please run the first script to download images.")
    else:
        root = tk.Tk()
        app = ImageFilterApp(root)
        root.mainloop()