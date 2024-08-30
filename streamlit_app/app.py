import streamlit as st
import os
import sys
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_mapping import create_database, get_objects, map_data, generate_output_table, update_extracted_text_for_master
from models.segmentation_model import load_model, segment_image
from models.identification_model import extract_identify_and_store_objects, load_identification_model
from models.text_extraction_model import extract_text_from_single_image
from models.summarization_model import update_captions_for_master
from utils.visualization import create_final_output

# Ensure the database is created and migrated once at the start
db_path = 'data/database.sqlite'
create_database(db_path)

# Load models
@st.cache_resource
def load_models():
    return load_model(), load_identification_model()

segmentation_model, identification_model = load_models()

# Set up the main navigation options
st.sidebar.title('Navigation')
option = st.sidebar.radio(
    "Go to",
    ('Segmentation', 'Object Extraction and Identification', 'Text Extraction', 'Summarization', 'Data Mapping', 'Output Generation')
)

st.title('Image Processing Pipeline')

def display_image_centered(image_path, caption):
    image = Image.open(image_path)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.write("")
    with col2:
        st.image(image, caption=caption, use_column_width=True)
    with col3:
        st.write("")

if option == "Segmentation":
    st.title('Image Segmentation')
    
    uploaded_file = st.file_uploader("Choose an image for segmentation...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_path = os.path.join('data', 'input_images', uploaded_file.name)
        output_dir = 'data/segmented_objects'

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        display_image_centered(file_path, 'Uploaded Image')
        
        if st.button('Segment Image'):
            with st.spinner('Processing image...'):
                image, masks, boxes, labels = segment_image(segmentation_model, file_path)
                st.success('Image segmentation completed.')
                # Visualization and results display
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.imshow(image)
                for i, (mask, box) in enumerate(zip(masks, boxes)):
                    color = np.random.rand(3)
                    masked = np.ma.masked_where(mask < 0.5, mask)
                    ax.imshow(masked, alpha=0.4, cmap=plt.colormaps.get_cmap('jet'), interpolation='none')
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
        
        display_image_centered(temp_file_path, 'Uploaded Image')
        
        if st.button('Extract and Identify Objects'):
            try:
                with st.spinner('Extracting and identifying objects...'):
                    # First, generate master_id by processing the image
                    master_id, object_data = extract_identify_and_store_objects(temp_file_path, output_dir, db_path, identification_model)
                    
                    # Now rename the file using the master_id
                    master_id_file_path = os.path.join('data', 'input_images', f'{master_id}.jpg')
                    os.rename(temp_file_path, master_id_file_path)
                    
                    st.success('Object extraction and identification completed.')

                    st.write(f"Processed image with master ID: {master_id}")
                    st.write(f"Extracted and identified {len(object_data)} unique objects")
                    
                    # Display the extracted and identified objects
                    for obj in object_data:
                        object_image_path = os.path.join(output_dir, obj['filename'])
                        display_image_centered(object_image_path, f"Object {obj['object_id']}: {obj['identification']}")
                    
                    # Display metadata table
                    st.subheader('Extracted and Identified Object Metadata')
                    df = pd.DataFrame(object_data)
                    st.table(df.iloc[:, :-2])

                    # Fetch from database
                    db_objects = get_objects(db_path, master_id)
                    st.subheader('Objects from Database')
                    db_df = pd.DataFrame(db_objects, columns=['Object ID', 'Master ID', 'Filename', 'Label', 'Identification', 'extracted_text', 'summary'])
                    st.table(db_df.iloc[:, :-2])
            except Exception as e:
                st.error(f"An error occurred during object extraction and identification: {str(e)}")

elif option == "Text Extraction":
    st.title('Image Upload and Text Extraction')
    
    # Fetch all master_ids from the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT master_id FROM objects")
    master_ids = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    selected_master_id = st.selectbox("Select a Master ID", master_ids)

    # Construct the image path using the master_id
    image_path = os.path.join('data', 'input_images', f'{selected_master_id}.jpg')
    
    if os.path.exists(image_path):
        display_image_centered(image_path, 'Image for Selected Master ID')
    else:
        st.info("Image not found for the selected Master ID.")
    
    if st.button('Extract Text'):
        try:
            with st.spinner('Extracting text from image...'):
                extracted_text = extract_text_from_single_image(image_path)
                
                st.write('Text extraction completed.')
                
                if extracted_text:
                    st.subheader('Extracted Text:')
                    st.success(extracted_text)
                    
                    # Update the database with the extracted text for all rows associated with the selected master_id
                    update_extracted_text_for_master(db_path, selected_master_id, extracted_text)
                    st.write(f'Text has been successfully stored in the database for master ID {selected_master_id}.')
                    
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
    
    # Construct the image path using the master_id
    image_path = os.path.join('data', 'input_images', f'{selected_master_id}.jpg')
    
    if os.path.exists(image_path):
        display_image_centered(image_path, 'Image for Selected Master ID')
    else:
        st.info("Image not found for the selected Master ID.")

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
    
    # Construct the image path using the master_id
    image_path = os.path.join('data', 'input_images', f'{selected_master_id}.jpg')
    
    if os.path.exists(image_path):
        display_image_centered(image_path, 'Image for Selected Master ID')
    else:
        st.info("Image not found for the selected Master ID.")
    
    if st.button('Map Data'):
        try:
            with st.spinner('Mapping data...'):
                # Map the data from the database to the final output format
                final_data = map_data(db_path, selected_master_id)
                st.success('Data mapping completed.')
                
                st.subheader('Final Mapped Data')
                st.json(final_data)
        except Exception as e:
            st.error(f"An error occurred during data mapping: {str(e)}")

elif option == "Output Generation":
    st.title('Output Generation')
    
    # Fetch all master_ids from the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT master_id FROM objects")
    master_ids = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    selected_master_id = st.selectbox("Select a Master ID", master_ids)
    
    # Construct the image path using the master_id
    image_path = os.path.join('data', 'input_images', f'{selected_master_id}.jpg')
    
    if os.path.exists(image_path):
        display_image_centered(image_path, 'Image for Selected Master ID')
    else:
        st.info("Image not found for the selected Master ID.")
    
    if st.button('Generate Final Output'):
        try:
            with st.spinner('Generating final output...'):
                final_output_path, mapped_data = create_final_output(image_path, db_path, selected_master_id)
                
                st.success('Final output generated.')
                st.image(final_output_path, caption='Final Output', use_column_width=True)
                st.subheader('Mapped Data')
                st.json(mapped_data)
                
                # Generate output table
                output_table = generate_output_table(db_path, selected_master_id)
                
                st.subheader('Summary Table')
                
                # Display the table with images
                for _, row in output_table.iterrows():
                    # Create two columns: one for image, one for data
                    image_col, data_col = st.columns([1, 3])
                    
                    # Display the object image
                    object_image_path = os.path.join('data', 'output', row['filename'])
                    if os.path.exists(object_image_path):
                        image_col.image(object_image_path, use_column_width=True)
                    else:
                        image_col.write("Image not found")
                    
                    # Display other columns
                    for column, value in row.items():
                        if column != 'filename':
                            data_col.write(f"**{column}:** {value}")
                    
                    st.write("---")
                
        except Exception as e:
            st.error(f"An error occurred during output generation: {str(e)}")
            st.error("Error details:", exc_info=True)  # This will print more detailed error information
st.sidebar.title('About')
st.sidebar.info('This app demonstrates an image processing pipeline that segments objects, identifies them, extracts text, and summarizes attributes.')