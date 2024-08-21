# AI Pipeline for Image Segmentation and Object Analysis

## Hosted Link
```bash
https://image-segmentation-and-object-analysis.streamlit.app/

```
## Project Overview

This project is an AI-powered pipeline designed for image segmentation and object identification. The pipeline segments objects within images and identifies them using state-of-the-art deep learning models. It also includes the ability to extract and store segmented objects, generate textual descriptions, and present the results through an interactive web application built with Streamlit.

The project is structured into multiple steps, each designed to handle a specific task in the pipeline, from image segmentation to object identification and description generation. The end goal is to provide an end-to-end solution for analyzing images and extracting meaningful insights.

## Features

- **Image Segmentation**: Utilizes the Mask R-CNN model to segment objects from input images.
- **Object Identification**: Implements pre-trained models such as YOLO and Faster R-CNN to identify objects within the segmented images.
- **Textual Description Generation**: Generates descriptions for identified objects using models like CLIP.
- **Database Integration**: Stores metadata and object details in an SQLite database for efficient data management.
- **Interactive Web Interface**: A user-friendly web application built with Streamlit allows users to upload images, perform segmentation, identification, and view results in real-time.

## Demo
Visit this link for demo, Working of website and model.
```
https://youtu.be/VwykU0esElY
```
[![Watch the video](https://img.youtube.com/vi/VwykU0esElY/maxresdefault.jpg)](https://www.youtube.com/watch?v=VwykU0esElY)


## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/sonalrajsr/sonal-wasserstoff-AiInternTask.git
```
2. Create a virtual environment (optional but recommended) & activate:
```bash
python -m venv <name_of_virtual_environment>
```
3. Install the required packages:
```bash
pip install -r requirements.txt
```
4. Download and set up the pre-trained models (if applicable):

5. Run app
```bash
streamlit run streamlit_app\app.py
```
   
## Models and Tasks

### 1. Mask R-CNN (maskrcnn_resnet50_fpn)
- **Task**: Image Segmentation
- **Purpose**: Segments objects in the input image, providing masks, bounding boxes, and labels for detected objects.

### 2. VGG16
- **Task**: Object Identification
- **Purpose**: Identifies and classifies the objects extracted from the segmentation step.

### 3. BLIP (Salesforce/blip-image-captioning-base)
- **Task**: Image Captioning / Summarization
- **Purpose**: Generates captions or summaries for the extracted objects.

### 4. EasyOCR
- **Task**: Text Extraction
- **Purpose**: Extracts any text present in the segmented objects or images.

## Project Structure
```markdown
project_root/
│
├── data/
│   ├── input_images/              # Directory for input images
│   ├── segmented_objects/         # Directory to save segmented object images
│   └── output/                    # Directory for output images and tables
│
├── models/
│   ├── segmentation_model.py      # Script for segmentation model
│   ├── identification_model.py    # Script for object identification model
│   ├── text_extraction_model.py   # Script for text/data extraction model
│   └── summarization_model.py     # Script for summarization model
│
├── utils/
│   ├── preprocessing.py           # Script for preprocessing functions
│   ├── postprocessing.py          # Script for postprocessing functions
│   ├── data_mapping.py            # Script for data mapping functions
│   └── visualization.py           # Script for visualization functions
```
- **app.py**: The main application file integrating all steps of the pipeline and serving the Streamlit interface.
- **segmentation.py**: Contains the code for segmenting images using Mask R-CNN.
- **identification.py**: Handles object identification using pre-trained models.
- **mapping.py**: Maps identified objects to their descriptions.
- **data/**: Directory for storing input images, segmented objects, and the SQLite database.
- **requirements.txt**: Lists all dependencies required to run the project.

