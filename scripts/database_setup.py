import sqlite3

def setup_database(db_name='images.db'):
    """
    Creates all necessary tables for the project.
    """
    conn = sqlite3.connect(db_name)
    c = conn.cursor()

    # Main table to index all images and track their processing state.
    # The 'filepath' column will store the RELATIVE path.
    c.execute('''
        CREATE TABLE IF NOT EXISTS image_index (
            id TEXT PRIMARY KEY,
            filepath TEXT UNIQUE NOT NULL,
            is_processed INTEGER DEFAULT 0 NOT NULL
        )
    ''')
    
    # A single table to store all calculated similarity features.
    c.execute('''
        CREATE TABLE IF NOT EXISTS image_characteristics (
            image_id TEXT PRIMARY KEY,
            feature_vector BLOB,
            embedding BLOB,
            fft_hash TEXT,
            FOREIGN KEY(image_id) REFERENCES image_index(id)
        )
    ''')

    conn.commit()
    conn.close()

if __name__ == '__main__':
    print("Setting up the database...")
    setup_database()
    print("Database setup complete.")