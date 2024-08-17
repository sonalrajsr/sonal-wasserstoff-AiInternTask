import sqlite3

def migrate_database(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if the identification column exists, if not, add it
    cursor.execute("PRAGMA table_info(objects)")
    columns = [column[1] for column in cursor.fetchall()]
    if 'identification' not in columns:
        cursor.execute('ALTER TABLE objects ADD COLUMN identification TEXT')
        print("Added 'identification' column to the objects table.")
    else:
        print("'identification' column already exists in the objects table.")
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    db_path = 'data/database.sqlite'  # Update this path if necessary
    migrate_database(db_path)