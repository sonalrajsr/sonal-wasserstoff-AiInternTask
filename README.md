# AI Pipeline for Image Segmentation and Object Analysis

## Project Overview

This project implements an AI pipeline that processes input images to segment, identify, and analyze objects within them. The pipeline outputs a summary table with mapped data for each object in the image.

## Features

- Image segmentation
- Object extraction and storage
- Object identification
- Text/data extraction from objects
- Object attribute summarization
- Data mapping
- Output generation with annotated images and summary tables
- Streamlit UI for easy testing and visualization

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/sonalrajsr/sonal-wasserstoff-AiInternTask.git
```
3. Create a virtual environment (optional but recommended):
```bash
python -m venv <name_of_virtual_environment>
```
5. Install the required packages:
```bash
pip install -r requirements.txt
```
7. Download and set up the pre-trained models (if applicable):

## Usage

1. To run the Streamlit app:
2. To use the pipeline programmatically:
```python
from models.segmentation_model import segment_image
from models.identification_model import identify_objects
# Import other necessary modules

# Use the pipeline functions
segmented_objects = segment_image(input_image)
identified_objects = identify_objects(segmented_objects)
# Continue with other pipeline steps
```
       
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
