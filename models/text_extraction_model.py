import easyocr
import os
import sqlite3
import streamlit as st
def load_text_extraction_model():
    try:
        reader = easyocr.Reader(['en'])  # Initialize EasyOCR
        st.write("EasyOCR model loaded successfully.")  # Confirmation message
        return reader
    except Exception as e:
        st.error(f"Failed to load EasyOCR model: {str(e)}")

def extract_text_from_image(reader, image_path):
    # Read text from the image
    result = reader.readtext(image_path)
    
    # Extract text from the result
    extracted_text = ' '.join([item[1] for item in result])
    
    return extracted_text

def extract_text_from_objects(output_dir, db_path):
    reader = load_text_extraction_model()
    
    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if the extracted_text column exists, if not, add it
    cursor.execute("PRAGMA table_info(objects)")
    columns = [column[1] for column in cursor.fetchall()]
    if 'extracted_text' not in columns:
        cursor.execute('ALTER TABLE objects ADD COLUMN extracted_text TEXT')
    
    # Fetch all objects from the database
    cursor.execute('SELECT object_id, filename FROM objects')
    objects = cursor.fetchall()
    
    for object_id, filename in objects:
        image_path = os.path.join(output_dir, filename)
        
        if os.path.exists(image_path):
            extracted_text = extract_text_from_image(reader, image_path)
            
            # Update the database with the extracted text
            cursor.execute('''
            UPDATE objects
            SET extracted_text = ?
            WHERE object_id = ?
            ''', (extracted_text, object_id))
    
    # Commit changes and close the connection
    conn.commit()
    conn.close()

    print("Text extraction completed and database updated.")

# Function to extract text from a single image (for use in the Streamlit app)
def extract_text_from_single_image(image_path):
    reader = load_text_extraction_model()
    return extract_text_from_image(reader, image_path)