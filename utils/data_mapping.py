import sqlite3
import json
import pandas as pd

def create_database(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS objects (
        object_id TEXT PRIMARY KEY,
        master_id TEXT,
        filename TEXT,
        label INTEGER,
        identification TEXT,
        extracted_text TEXT,
        summary TEXT
    )
    ''')
    conn.commit()
    conn.close()


def get_objects(db_path, master_id):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT * FROM objects WHERE master_id = ?
    ''', (master_id,))
    
    objects = cursor.fetchall()
    conn.close()
    return objects


############################################################################################################


# summary TEXT

def map_data(db_path, master_id):
    conn = sqlite3.connect(db_path)
    
    # Fetch all data for the given master_id
    query = '''
    SELECT object_id, master_id, filename, label, identification, extracted_text, summary
    FROM objects
    WHERE master_id = ?
    '''
    df = pd.read_sql_query(query, conn, params=(master_id,))
    conn.close()
    
    # Convert DataFrame to JSON
    json_data = df.to_json(orient='records')
    
    return json.loads(json_data)

def generate_output_table(db_path, master_id):
    conn = sqlite3.connect(db_path)
    
    query = '''
    SELECT filename, object_id, identification, extracted_text, summary
    FROM objects
    WHERE master_id = ?
    '''
    df = pd.read_sql_query(query, conn, params=(master_id,))
    conn.close()
    
    return df


def update_extracted_text_for_master(db_path, master_id, extracted_text):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
    UPDATE objects
    SET extracted_text = ?
    WHERE master_id = ?
    ''', (extracted_text, master_id))
    conn.commit()
    conn.close()
