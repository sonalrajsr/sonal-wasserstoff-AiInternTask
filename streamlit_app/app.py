import streamlit as st
import os
import sys
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_mapping import create_database, get_objects
from models.segmentation_model import load_model, segment_image
from models.identification_model import extract_identify_and_store_objects
# from models.text_extraction_model import extract_text
# from models.summarization_model import summarize_attributes
# from utils.data_mapping import map_data
# from utils.visualization import generate_output_image

# Ensure the database is created and migrated once at the start
db_path = 'data/database.sqlite'
create_database(db_path)

# Load models
@st.cache_resource
def load_models():
    return load_model()

segmentation_model = load_models()

# Set up the main navigation options
st.sidebar.title('Navigation')
option = st.sidebar.radio(
    "Go to",
    ('Segmentation', 'Object Extraction and Identification', 'Text Extraction', 'Summarization', 'Data Mapping', 'Output Generation')
)

st.title('Image Processing Pipeline')

if option == "Segmentation":
    st.title('Image Segmentation')
    
    uploaded_file = st.file_uploader("Choose an image for segmentation...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_path = os.path.join('data', 'input_images', uploaded_file.name)
        output_dir = 'data\segmented_objects'

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        
        if st.button('Segment Image'):
            with st.spinner('Processing image...'):
                image, masks, boxes, labels = segment_image(segmentation_model, file_path)
                st.success('Image segmentation completed.')
                # Visualization and results display
                fig, ax = plt.subplots()
                ax.imshow(image)
                for i, (mask, box) in enumerate(zip(masks, boxes)):
                    color = np.random.rand(3)
                    masked = np.ma.masked_where(mask < 0.5, mask)
                    ax.imshow(masked, alpha=0.4, cmap=plt.cm.get_cmap('jet'), interpolation='none')
                    x1, y1, x2, y2 = box.astype(int)
                    ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2))
                    ax.text(x1, y1, f"Object {i+1}", bbox=dict(facecolor=color, alpha=0.5), fontsize=8, color='white')

                    # Save the segmented object
                    segmented_object = image.crop((x1, y1, x2, y2))
                    object_path = os.path.join(output_dir, f"segmented_object_{i+1}.png")
                    segmented_object.save(object_path)
                st.pyplot(fig)

elif option == "Object Extraction and Identification":
    st.title('Object Extraction and Identification')
    
    uploaded_file = st.file_uploader("Choose an image for object extraction and identification...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_path = os.path.join('data', 'input_images', uploaded_file.name)
        output_dir = 'data/output'

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        
        if st.button('Extract and Identify Objects'):
            try:
                with st.spinner('Extracting and identifying objects...'):
                    master_id, object_data = extract_identify_and_store_objects(file_path, output_dir, db_path)
                    
                    st.success('Object extraction and identification completed.')

                    st.write(f"Processed image with master ID: {master_id}")
                    st.write(f"Extracted and identified {len(object_data)} unique objects")
                    
                    # Display the extracted and identified objects
                    for obj in object_data:
                        object_image_path = os.path.join(output_dir, obj['filename'])
                        st.image(object_image_path, caption=f"Object {obj['object_id']}: {obj['identification']}", use_column_width=True)
                    
                    # Display metadata table
                    st.subheader('Extracted and Identified Object Metadata')
                    df = pd.DataFrame(object_data)
                    st.table(df)

                    # Fetch from database
                    db_objects = get_objects(db_path, master_id)
                    st.subheader('Objects from Database')
                    db_df = pd.DataFrame(db_objects, columns=['Object ID', 'Master ID', 'Filename', 'Label', 'Identification'])
                    st.table(db_df)
            except Exception as e:
                st.error(f"An error occurred during object extraction and identification: {str(e)}")
                st.error("Please check if 'imagenet_classes.txt' is present in the project root directory.")
st.sidebar.title('About')
st.sidebar.info('This app demonstrates an image processing pipeline that segments objects, identifies them, extracts text, and summarizes attributes.')

