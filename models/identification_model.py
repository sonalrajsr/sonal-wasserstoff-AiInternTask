import os
import uuid
import sqlite3
from PIL import Image
import numpy as np
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch
from torchvision.models import vgg16, VGG16_Weights
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from models.segmentation_model import segment_image

def load_model():
    model = maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

def load_identification_model():
    model = vgg16(weights=VGG16_Weights.DEFAULT)
    model.eval()
    return model

def preprocess_image(image):
    # Use the preprocessing pipeline recommended for VGG16
    transform = Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

def identify_object(model, image):
    image_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(image_tensor)
    
    weights = VGG16_Weights.DEFAULT
    categories = weights.meta["categories"]
    
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top_prob, top_catid = torch.topk(probabilities, 1)
    
    return categories[top_catid[0]]

def extract_identify_and_store_objects(image_path, output_dir, db_path, identification_model, max_objects=5):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    # Load the segmentation model
    segmentation_model = load_model()
    # Load the identification model
    identification_model = load_identification_model()
    # Perform segmentation
    image, masks, boxes, labels = segment_image(segmentation_model, image_path)
    # Generate a master ID for the original image
    master_id = str(uuid.uuid4())
    # Convert PIL Image to numpy array
    image_np = np.array(image)
    # Extract, identify, and save each object
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
        # Identify the object
        identification = identify_object(identification_model, background)
        # Save the object image
        object_filename = f"{object_id}.png"
        object_path = os.path.join(output_dir, object_filename)
        background.save(object_path)
        # Store object metadata
        object_data.append({
            'object_id': object_id,
            'master_id': master_id,
            'filename': object_filename,
            'label': label.item(),
            'identification': identification,
            'extracted_text': None, 
            'summary': None 
        })
    # Store metadata in SQLite database
    store_metadata(db_path, object_data)
    return master_id, object_data

def store_metadata(db_path, object_data):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Insert object data
    cursor.executemany('''
    INSERT INTO objects (object_id, master_id, filename, label, identification, extracted_text, summary)
    VALUES (:object_id, :master_id, :filename, :label, :identification, :extracted_text, :summary)
    ''', object_data)
    conn.commit()
    conn.close()