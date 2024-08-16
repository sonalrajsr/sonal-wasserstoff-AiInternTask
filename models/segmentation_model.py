import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import io

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

def visualize_segmentation_streamlit(image, masks, boxes):
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    
    for i, (mask, box) in enumerate(zip(masks, boxes)):
        color = np.random.rand(3)
        
        # Display segmentation mask
        masked = np.ma.masked_where(mask < 0.5, mask)
        plt.imshow(masked, alpha=0.4, cmap=plt.cm.get_cmap('jet'), interpolation='none')
        
        # Display bounding box
        x1, y1, x2, y2 = box.astype(int)
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2))
        plt.gca().text(x1, y1, f"Object {i+1}", bbox=dict(facecolor=color, alpha=0.5), fontsize=8, color='white')

    plt.axis('off')
    plt.tight_layout()
    
    # Save the plot to a bytes object and display in Streamlit
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return buf

def main():
    model = load_model()
    image_path = "data\input_images\bigstock-Kids-Play-Football-Child-At-S-250398277.jpg"  # Replace with your image path
    image, masks, boxes, labels = segment_image(model, image_path)
    visualize_segmentation_streamlit(image, masks, boxes)

if __name__ == "__main__":
    main()