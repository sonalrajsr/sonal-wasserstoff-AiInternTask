import os
import uuid
import sqlite3
from PIL import Image
import numpy as np
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F

def load_model():
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
    masks = prediction[0]['masks'].squeeze().detach().cpu().numpy()
    boxes = prediction[0]['boxes'].detach().cpu().numpy()
    labels = prediction[0]['labels'].detach().cpu().numpy()
    scores = prediction[0]['scores'].detach().cpu().numpy()
    # Filter out low-confidence predictions
    threshold = 0.5
    mask_indices = np.where(scores > threshold)[0]
    return image, masks[mask_indices], boxes[mask_indices], labels[mask_indices]

def extract_and_store_objects(image_path, output_dir, db_path, max_objects=5):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    # Load the segmentation model
    model = load_model()
    # Perform segmentation
    image, masks, boxes, labels = segment_image(model, image_path)
    # Generate a master ID for the original image
    master_id = str(uuid.uuid4())
    # Convert PIL Image to numpy array
    image_np = np.array(image)
    # Extract and save each object
    object_data = []
    for i, (mask, box, label) in enumerate(zip(masks, boxes, labels)):
        if len(object_data) >= max_objects:
            break
        
        # Generate a unique ID for the object
        object_id = str(uuid.uuid4())
        # Extract the object using the mask
        object_mask = mask > 0.5
        object_image = image_np * object_mask[:, :, np.newaxis]
        # Convert to PIL Image
        object_pil = Image.fromarray(object_image.astype('uint8'), 'RGB')
        # Create a white background
        background = Image.new('RGB', object_pil.size, (255, 255, 255))
       
        # Paste the object onto the white background
        background.paste(object_pil, (0, 0), Image.fromarray((object_mask * 255).astype('uint8')))
        # Save the object image
        object_filename = f"{object_id}.png"
        object_path = os.path.join(output_dir, object_filename)
        background.save(object_path)
        # Store object metadata
        object_data.append({
            'object_id': object_id,
            'master_id': master_id,
            'filename': object_filename,
            'label': label.item()
        })
    # Store metadata in SQLite database
    store_metadata(db_path, object_data)
    return master_id, object_data

def store_metadata(db_path, object_data):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Insert object data
    cursor.executemany('''
    INSERT INTO objects (object_id, master_id, filename, label)
    VALUES (:object_id, :master_id, :filename, :label)
    ''', object_data)
    conn.commit()
    conn.close()
