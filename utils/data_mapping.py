import sqlite3

def create_database(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS objects (
        object_id TEXT PRIMARY KEY,
        master_id TEXT,
        filename TEXT,
        label INTEGER,
        identification TEXT
    )
    ''')
    
    # Check if the identification column exists, if not, add it
    cursor.execute("PRAGMA table_info(objects)")
    columns = [column[1] for column in cursor.fetchall()]
    if 'identification' not in columns:
        cursor.execute('ALTER TABLE objects ADD COLUMN identification TEXT')
    
    conn.commit()
    conn.close()
    
def insert_objects(db_path, object_data):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.executemany('''
    INSERT INTO objects (object_id, master_id, filename, label)
    VALUES (:object_id, :master_id, :filename, :label)
    ''', object_data)
    
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
