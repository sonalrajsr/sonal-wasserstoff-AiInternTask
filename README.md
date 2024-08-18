# AI Pipeline for Image Segmentation and Object Analysis

## Project Overview

This project is an AI-powered pipeline designed for image segmentation and object identification. The pipeline segments objects within images and identifies them using state-of-the-art deep learning models. It also includes the ability to extract and store segmented objects, generate textual descriptions, and present the results through an interactive web application built with Streamlit.

The project is structured into multiple steps, each designed to handle a specific task in the pipeline, from image segmentation to object identification and description generation. The end goal is to provide an end-to-end solution for analyzing images and extracting meaningful insights.

## Features

- **Image Segmentation**: Utilizes the Mask R-CNN model to segment objects from input images.
- **Object Identification**: Implements pre-trained models such as YOLO and Faster R-CNN to identify objects within the segmented images.
- **Textual Description Generation**: Generates descriptions for identified objects using models like CLIP.
- **Database Integration**: Stores metadata and object details in an SQLite database for efficient data management.
- **Interactive Web Interface**: A user-friendly web application built with Streamlit allows users to upload images, perform segmentation, identification, and view results in real-time.


## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/sonalrajsr/sonal-wasserstoff-AiInternTask.git
```
2. Create a virtual environment (optional but recommended):
```bash
python -m venv <name_of_virtual_environment>
```
3. Install the required packages:
```bash
pip install -r requirements.txt
```
4. Download and set up the pre-trained models (if applicable):

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
## Project Structure

- **app.py**: The main application file integrating all steps of the pipeline and serving the Streamlit interface.
- **segmentation.py**: Contains the code for segmenting images using Mask R-CNN.
- **identification.py**: Handles object identification using pre-trained models.
- **mapping.py**: Maps identified objects to their descriptions.
- **data/**: Directory for storing input images, segmented objects, and the SQLite database.
- **requirements.txt**: Lists all dependencies required to run the project.

