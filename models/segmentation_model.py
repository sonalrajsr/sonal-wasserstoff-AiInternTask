import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np

def load_model():
    # Load pre-trained Mask R-CNN model
    model = maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

def segment_image(model, image_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image).unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        prediction = model(image_tensor)

    # Extract masks, boxes, and labels
    masks = prediction[0]['masks'].squeeze().detach().numpy()
    boxes = prediction[0]['boxes'].detach().numpy()
    labels = prediction[0]['labels'].detach().numpy()
    scores = prediction[0]['scores'].detach().numpy()

    # Filter out low-confidence predictions
    threshold = 0.5
    mask_indices = np.where(scores > threshold)[0]

    return image, masks[mask_indices], boxes[mask_indices], labels[mask_indices]
