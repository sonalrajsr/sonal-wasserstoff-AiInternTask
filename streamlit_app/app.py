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
from models.identification_model import extract_and_store_objects
# from models.text_extraction_model import extract_text
# from models.summarization_model import summarize_attributes
# from utils.data_mapping import map_data
# from utils.visualization import generate_output_image

# Ensure the database is created once at the start
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
    ('Segmentation', 'Identification', 'Text Extraction', 'Summarization', 'Data Mapping', 'Output Generation')
)

st.title('Image Processing Pipeline')

if option == "Segmentation":
    st.title('Image Segmentation')
    
    uploaded_file = st.file_uploader("Choose an image for segmentation...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_path = os.path.join('data', 'input_images', uploaded_file.name)
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
                st.pyplot(fig)
                # ...

elif option == "Identification":
    st.title('Object Identification')
    
    uploaded_file = st.file_uploader("Choose an image for identification...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_path = os.path.join('data', 'input_images', uploaded_file.name)
        output_dir = 'data/output'

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        
        if st.button('Identify Objects'):
            with st.spinner('Identifying objects...'):
                master_id, object_data = extract_and_store_objects(file_path, output_dir, db_path)
                st.success('Object identification completed.')

                st.write(f"Processed image with master ID: {master_id}")
                st.write(f"Extracted {len(object_data)} unique objects")
                
                # Display the extracted objects
                for obj in object_data:
                    object_image_path = os.path.join(output_dir, obj['filename'])
                    st.image(object_image_path, caption=f"Object {obj['object_id']}", use_column_width=True)
                    
                # Display metadata table
                st.subheader('Extracted Object Metadata')
                df = pd.DataFrame(object_data)
                st.table(df)

                # Fetch from database
                db_objects = get_objects(db_path, master_id)
                st.subheader('Objects from Database')
                db_df = pd.DataFrame(db_objects, columns=['Object ID', 'Master ID', 'Filename', 'Label'])
                st.table(db_df)

            # elif option == 'Text Extraction':
            #     # Step 4: Text/Data Extraction from Objects
            #     extracted_text = extract_text(image, boxes)
            #     st.success('Text extraction completed.')
                
            #     st.subheader('Extracted Text')
            #     st.write(extracted_text)

            # elif option == 'Summarization':
            #     # Step 5: Summarize Object Attributes
            #     summaries = summarize_attributes(identified_objects, extracted_text)
            #     st.success('Object attribute summarization completed.')
                
            #     st.subheader('Summarized Attributes')
            #     st.write(summaries)

            # elif option == 'Data Mapping':
            #     # Step 6: Data Mapping
            #     mapped_data = map_data(identified_objects, extracted_text, summaries)
            #     st.success('Data mapping completed.')
                
            #     st.subheader('Mapped Data')
            #     df = pd.DataFrame(mapped_data)
            #     st.table(df)

            # elif option == 'Output Generation':
            #     # Step 7: Output Generation
            #     output_image = generate_output_image(image, masks, boxes, mapped_data)
                
            #     st.subheader('Processed Image')
            #     st.image(output_image, caption='Processed Image', use_column_width=True)
                
            #     # Save output
            #     output_path = os.path.join('data', 'output', f'output_{uploaded_file.name}')
            #     output_image.save(output_path)
            #     st.success(f'Output saved to {output_path}')

st.sidebar.title('About')
st.sidebar.info('This app demonstrates an image processing pipeline that segments objects, identifies them, extracts text, and summarizes attributes.')

