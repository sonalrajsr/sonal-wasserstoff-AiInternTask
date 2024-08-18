import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import sqlite3

def load_summarization_model():
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    return tokenizer, model

def generate_summary(tokenizer, model, text, max_length=100):
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def summarize_object_attributes(db_path, output_dir):
    tokenizer, model = load_summarization_model()
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if the summary column exists, if not, add it
    cursor.execute("PRAGMA table_info(objects)")
    columns = [column[1] for column in cursor.fetchall()]
    if 'summary' not in columns:
        cursor.execute('ALTER TABLE objects ADD COLUMN summary TEXT')
    
    # Fetch all objects from the database
    cursor.execute('SELECT object_id, identification, extracted_text FROM objects')
    objects = cursor.fetchall()
    
    summaries = []
    
    for object_id, identification, extracted_text in objects:
        # Combine identification and extracted text for summarization
        full_text = f"Object: {identification}. Extracted text: {extracted_text}"
        
        # Generate summary
        summary = generate_summary(tokenizer, model, full_text)
        
        # Update the database with the summary
        cursor.execute('''
        UPDATE objects
        SET summary = ?
        WHERE object_id = ?
        ''', (summary, object_id))
        
        summaries.append({
            'object_id': object_id,
            'summary': summary
        })
    
    # Commit changes and close the connection
    conn.commit()
    conn.close()
    
    return summaries

# Function to summarize a single object (for use in the Streamlit app)
def summarize_single_object(identification, extracted_text):
    tokenizer, model = load_summarization_model()
    full_text = f"Object: {identification}. Extracted text: {extracted_text}"
    summary = generate_summary(tokenizer, model, full_text)
    return summary