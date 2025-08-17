import sqlite3
import pandas as pd
import numpy as np

def create_connection(db_file="images.db"):
    """ Create a database connection to the SQLite database. """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except sqlite3.Error as e:
        print(e)
    return conn

def load_all_features_to_dataframe(db_name='images.db'):
    """
    Loads all pre-calculated features from the database into a pandas DataFrame.
    """
    print("Loading pre-calculated features from the database...")
    conn = create_connection(db_name)
    
    query = """
    SELECT
        idx.filepath,
        char.feature_vector,
        char.embedding,
        char.fft_hash
    FROM
        image_characteristics char
    JOIN
        image_index idx ON char.image_id = idx.id
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()

    print("Deserializing and preparing data for fast searching...")
    # Reshape the color vector back to a 2D array with 4 columns.
    df['feature_vector'] = df['feature_vector'].apply(
        lambda x: np.frombuffer(x, dtype=np.float32).reshape(-1, 6)
    )
    df['embedding'] = df['embedding'].apply(lambda x: np.frombuffer(x, dtype=np.float32))

    # --- NEW: Pre-process FFT hashes for vectorization ---
    # Convert the column of hash strings into a 2D NumPy array of integers (0s and 1s).
    # This is a one-time cost at startup that makes the search much faster.
    df['fft_hash_array'] = df['fft_hash'].apply(lambda x: np.array(list(x), dtype=np.int8))

    print(f"Successfully loaded and prepared {len(df)} feature sets.")
    return df
