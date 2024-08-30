import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from models.segmentation_model import segment_image, load_model
from utils.data_mapping import map_data, generate_output_table

def generate_output_image(image_path, db_path, master_id):
    # Load the segmentation model
    model = load_model()
    
    # Segment the image
    image, masks, boxes, labels = segment_image(model, image_path)
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Display the original image
    ax.imshow(image)
    
    # Get mapped data
    mapped_data = map_data(db_path, master_id)
    
    # Draw bounding boxes and labels
    for i, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = box.astype(int)
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        ax.text(x1, y1, f"Object {i+1}", bbox=dict(facecolor='red', alpha=0.5), fontsize=8, color='white')
    
    ax.axis('off')
    plt.tight_layout()
    
    # Save the annotated image
    output_image_path = 'data/output/annotated_image.png'
    plt.savefig(output_image_path)
    plt.close()
    
    return output_image_path, mapped_data

def generate_output_summary(db_path, master_id):
    # Generate the output table
    df = generate_output_table(db_path, master_id)
    
    # Create a summary string
    summary = "Object Summary:\n\n"
    for _, row in df.iterrows():
        summary += f"Object ID: {row['object_id']}\n"
        summary += f"Identification: {row['identification']}\n"
        summary += f"Extracted Text: {row['extracted_text']}\n"
        # summary += f"Summary: {row['summary']}\n\n"
    
    return summary

def create_final_output(image_path, db_path, master_id):
    # Generate annotated image and get mapped data
    annotated_image_path, mapped_data = generate_output_image(image_path, db_path, master_id)
    
    # Generate summary table
    summary = generate_output_summary(db_path, master_id)
    
    # Create the final output image with summary
    annotated_image = Image.open(annotated_image_path)
    width, height = annotated_image.size
    
    # Create a new image with extra space for the summary
    final_image = Image.new('RGB', (width, height + 400), color='white')
    final_image.paste(annotated_image, (0, 0))
    
    # Add summary text
    draw = ImageDraw.Draw(final_image)
    font = ImageFont.load_default()
    draw.text((10, height + 10), summary, fill='black', font=font)
    
    # Save the final output
    final_output_path = 'data/output/final_output.png'
    final_image.save(final_output_path)
    
    return final_output_path, mapped_data
