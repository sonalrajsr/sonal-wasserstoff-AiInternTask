import streamlit as st
import os
import sys
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import sqlite3
# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_mapping import create_database, get_objects
from models.segmentation_model import load_model, segment_image
from models.identification_model import extract_identify_and_store_objects
from models.text_extraction_model import extract_text_from_single_image
from models.summarization_model import update_captions_for_master
from utils.data_mapping import map_data, generate_output_table, update_extracted_text_for_master
from utils.visualization import create_final_output


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
        # Temporarily save the uploaded file
        temp_file_path = os.path.join('data', 'input_images', uploaded_file.name)
        output_dir = 'data/output'

        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        
        if st.button('Extract and Identify Objects'):
            try:
                with st.spinner('Extracting and identifying objects...'):
                    # First, generate master_id by processing the image
                    master_id, object_data = extract_identify_and_store_objects(temp_file_path, output_dir, db_path)
                    
                    # Now rename the file using the master_id
                    master_id_file_path = os.path.join('data', 'input_images', f'{master_id}.jpg')
                    os.rename(temp_file_path, master_id_file_path)
                    
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
                    db_df = pd.DataFrame(db_objects, columns=['Object ID', 'Master ID', 'Filename', 'Label', 'Identification', 'extracted_text', 'summary'])
                    st.table(db_df.iloc[:, :-2])
            except Exception as e:
                st.error(f"An error occurred during object extraction and identification: {str(e)}")
                st.error("Please check if 'imagenet_classes.txt' is present in the project root directory.")

elif option == "Text Extraction":
    st.title('Image Upload and Text Extraction')
    
    # Fetch all master_ids from the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT master_id FROM objects")
    master_ids = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    selected_master_id = st.selectbox("Select a Master ID", master_ids)

    uploaded_file = st.file_uploader("Choose an image for text extraction...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_path = os.path.join('data', 'input_images', uploaded_file.name)

        # Ensure input directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        
        if st.button('Extract Text'):
            try:
                with st.spinner('Extracting text from image...'):
                    extracted_text = extract_text_from_single_image(file_path)
                    
                    st.success('Text extraction completed.')
                    
                    if extracted_text:
                        st.subheader('Extracted Text:')
                        st.write(extracted_text)
                        
                        # Update the database with the extracted text for all rows associated with the selected master_id
                        update_extracted_text_for_master(db_path, selected_master_id, extracted_text)
                        st.success(f'Text has been successfully stored in the database for master ID {selected_master_id}.')
                        
                    else:
                        st.info("No text was extracted from the image.")
            except Exception as e:
                st.error(f"An error occurred during text extraction: {str(e)}")

elif option == "Summarization":
    st.title('Object Attribute Summarization')
    
    # Fetch all master_ids from the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT master_id FROM objects")
    master_ids = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    selected_master_id = st.selectbox("Select a Master ID", master_ids)
    
    if st.button('Summarize Object Attributes'):
        try:
            with st.spinner('Summarizing object attributes and generating captions...'):
                # Update captions for all images linked to the selected master_id
                update_captions_for_master(db_path, selected_master_id)
                
                # Fetch and display the updated object data
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT object_id, identification, extracted_text, summary FROM objects WHERE master_id = ?", (selected_master_id,))
                objects = cursor.fetchall()
                conn.close()
                
                st.success('Summarization completed.')
                
                st.subheader('Objects with Summaries')
                if objects:
                    columns = ['Object ID', 'Identification', 'Extracted Text', 'Summary']
                    df = pd.DataFrame(objects, columns=columns)
                    st.dataframe(df)
                else:
                    st.info("No objects found in the database.")
        except Exception as e:
            st.error(f"An error occurred during object attribute summarization: {str(e)}")

elif option == "Data Mapping":
    st.title('Data Mapping')
    
    # Fetch all master_ids from the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT master_id FROM objects")
    master_ids = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    selected_master_id = st.selectbox("Select a Master ID", master_ids)
    
    if st.button('Map Data'):
        mapped_data = map_data(db_path, selected_master_id)
        st.json(mapped_data)

elif option == "Output Generation":
    st.title('Output Generation')

    # Fetch all master_ids from the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT master_id FROM objects")
    master_ids = [row[0] for row in cursor.fetchall()]
    conn.close()

    # Unique key added to st.selectbox to prevent DuplicateWidgetID error
    selected_master_id = st.selectbox("Select a Master ID", master_ids, key="output_generation_master_id")

    if st.button('Generate Output'):
        # Construct the filename and path for the selected master_id
        filename = selected_master_id
        extensions = ['.jpg', '.jpeg', '.png']
        image_path = None

        # Try each extension to find a valid image file
        for ext in extensions:
            possible_path = os.path.join('data', 'input_images', f'{filename}{ext}')
            if os.path.isfile(possible_path):
                image_path = possible_path
                break

        if image_path:
            # Generate final output and mapped data
            final_output_path, mapped_data = create_final_output(image_path, db_path, selected_master_id)

            # Display the final output image
            st.image(final_output_path, caption='Final Output', use_column_width=True)

            # Display mapped data
            st.subheader('Mapped Data')
            st.json(mapped_data)

            # Display summary table
            st.subheader('Summary Table')
            df = generate_output_table(db_path, selected_master_id)
            st.table(df)

        else:
            st.error("No valid image file found for the selected Master ID.")

st.sidebar.title('About')
st.sidebar.info('This app demonstrates an image processing pipeline that segments objects, identifies them, extracts text, and summarizes attributes.')

