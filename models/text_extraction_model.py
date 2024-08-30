import easyocr
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

# Function to extract text from a single image (for use in the Streamlit app)
def extract_text_from_single_image(image_path):
    reader = load_text_extraction_model()
    return extract_text_from_image(reader, image_path)