import torch
from model import UNET
from weaviateUtils import compute_moments_and_hu, query_weaviate_hu_moments, getWeaviate
import albumentations as A
import cv2
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNET()

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Grayscale load
    image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)  # Match training size
    image = image.astype('float32') / 255.0  # Normalize to [0, 1]
    image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)  # (1, 1, 512, 512)
    return image

def predict(model, image_tensor, device):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor)
        print(f"Raw output shape: {output.shape}")
        print(f"Raw output range: {output.min().item()} to {output.max().item()}")
        prediction = torch.sigmoid(output)
        print(f"After sigmoid range: {prediction.min().item()} to {prediction.max().item()}")
        prediction = (prediction > 0.5).float() * 255 # Match training threshold logic
    return prediction

def postprocess_prediction(prediction):
    # Remove batch dimension and convert to numpy
    prediction = prediction.squeeze(0).cpu().numpy()
    # Convert from [C, H, W] to [H, W, C] if needed
    if prediction.shape[0] == 1:  # If single channel
        prediction = prediction[0]  # [H, W]
    return prediction

def visualize_prediction(original_image_path, prediction):
    # Read original image for comparison
    original = cv2.imread(original_image_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # Process prediction
    prediction = prediction.squeeze(0).cpu().numpy()
    if prediction.shape[0] == 1:  # If single channel
        prediction = prediction[0]  # [H, W]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Show original image
    ax1.imshow(original)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Show prediction
    ax2.imshow(prediction, cmap='gray')
    ax2.set_title('Predicted Edge Mask')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

# Define validation transforms
val_transforms = A.Compose([
    A.Resize(height=512, width=512)
])

client, collection = getWeaviate()

try:
    response = collection.query.fetch_objects()

    for o in response.objects:
        print(o.properties)

    model.load_state_dict(torch.load("edge_segmentation_model.pth"))
    model = model.to(device)
    model.eval()

    image_path = "../images/19fgtel_0.jpg"
    image_tensor = preprocess_image(image_path).to(device)
    prediction = predict(model, image_tensor, device)

    visualize_prediction(image_path, prediction)

finally:
    client.close()