from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import sqlite3
import os
# Load the BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image_path):
    # Load the image
    image = Image.open(image_path)
    
    # Process the image and generate a caption
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    
    # Decode the generated output
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def update_captions_for_master(db_path, master_id):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Fetch all image filenames associated with the master_id
    cursor.execute('''
    SELECT filename FROM objects WHERE master_id = ?
    ''', (master_id,))
    filenames = [row[0] for row in cursor.fetchall()]

    for filename in filenames:
        image_path = os.path.join('data', 'output', filename)
        if os.path.isfile(image_path):
            # Generate caption
            caption = generate_caption(image_path)
            # Update the database with the generated caption
            cursor.execute('''
            UPDATE objects
            SET summary = ?
            WHERE filename = ?
            ''', (caption, filename))
        else:
            print(f"Image file {filename} not found.")
    
    conn.commit()
    conn.close()
