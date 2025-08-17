import argparse
import os
import sys
import cv2
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import sqlite3
import glob

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.database import create_connection
# Import from the correct, isolated shazam similarity file
from src.shazam_similarity import get_shazam_fingerprints

def setup_temp_database(db_path):
    """Creates a temporary database file for a worker."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE shazam_fingerprints (
            hash TEXT NOT NULL, image_id TEXT NOT NULL, 
            anchor_y INTEGER NOT NULL, anchor_x INTEGER NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def worker_task(image_chunk, image_root_folder, worker_id):
    """
    A worker's complete task: process a chunk of images and write the results
    to a unique temporary database file.
    """
    temp_db_path = f'_temp_chunk_{worker_id}.db'
    setup_temp_database(temp_db_path)
    
    conn = sqlite3.connect(temp_db_path)
    c = conn.cursor()

    all_fingerprints_for_chunk = []
    for image_id, relative_path in image_chunk:
        full_path = os.path.join(image_root_folder, relative_path)
        image = cv2.imread(full_path)
        if image is not None:
            fingerprints, _ = get_shazam_fingerprints(image)
            all_fingerprints_for_chunk.extend(
                [(h, image_id, int(ay), int(ax)) for h, (ay, ax) in fingerprints]
            )

    if all_fingerprints_for_chunk:
        c.executemany("INSERT INTO shazam_fingerprints VALUES (?, ?, ?, ?)", all_fingerprints_for_chunk)
        conn.commit()
        
    conn.close()
    return len(all_fingerprints_for_chunk)

def run_shazam_preprocessing(image_root_folder, db_name='images.db', chunk_size=2000):
    """
    Orchestrates the Map-Reduce pipeline for fingerprinting.
    """
    conn = create_connection(db_name)
    c = conn.cursor()

    print("--- Clearing any old Shazam fingerprints ---")
    c.execute("DELETE FROM shazam_fingerprints")
    conn.commit()

    print("--- Finding all images to process ---")
    c.execute("SELECT id, filepath FROM image_index")
    images_to_process = c.fetchall()
    
    if not images_to_process:
        print("No images found in the index. Please run the discovery script first.")
        conn.close()
        return

    # --- SETUP PHASE: Divide the work into chunks ---
    num_images = len(images_to_process)
    chunks = [images_to_process[i:i + chunk_size] for i in range(0, num_images, chunk_size)]
    print(f"Divided {num_images} images into {len(chunks)} chunks of up to {chunk_size} images each.")

    # --- MAP PHASE: Process chunks in parallel ---
    print(f"\n--- Starting Map Phase: Processing {len(chunks)} chunks in parallel...")
    # Use 8 cores as requested
    with Pool(processes=8) as pool:
        # Create a list of arguments for each worker
        tasks = [(chunk, image_root_folder, i) for i, chunk in enumerate(chunks)]
        
        # --- THIS IS THE PROGRESS BAR YOU REQUESTED ---
        # The main process will track the completion of each chunk.
        results = list(tqdm(pool.starmap(worker_task, tasks), total=len(tasks), desc="Processing Chunks"))

    total_fingerprints = sum(results)
    print(f"--- Map Phase Complete: Generated a total of {total_fingerprints} fingerprints across all workers.")

    # --- REDUCE PHASE: Merge temporary databases ---
    print("\n--- Starting Reduce Phase: Merging temporary databases into the main database...")
    temp_db_files = glob.glob('_temp_chunk_*.db')
    
    c.execute("BEGIN TRANSACTION;")
    for temp_db in tqdm(temp_db_files, desc="Merging Chunks"):
        temp_conn = sqlite3.connect(temp_db)
        temp_c = temp_conn.cursor()
        
        temp_c.execute("SELECT * FROM shazam_fingerprints")
        fingerprints_from_chunk = temp_c.fetchall()
        if fingerprints_from_chunk:
            c.executemany("INSERT INTO shazam_fingerprints VALUES (?, ?, ?, ?)", fingerprints_from_chunk)
        
        temp_conn.close()
    
    c.execute("COMMIT;")
    print("--- Reduce Phase Complete ---")
    
    # --- CLEANUP PHASE ---
    print("\n--- Cleaning up temporary files ---")
    for temp_db in temp_db_files:
        os.remove(temp_db)
    print("Cleanup complete.")

    conn.close()
    print(f"\n--- Shazam preprocessing complete! Inserted {total_fingerprints} fingerprints in total. ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate and store Shazam-style fingerprints using a Map-Reduce pipeline.")
    parser.add_argument("image_folder", type=str, help="The CURRENT root folder of your image collection.")
    parser.add_argument("--db_name", type=str, default="images.db", help="The SQLite database file name.")
    parser.add_argument("--chunk_size", type=int, default=500, help="Number of images for each parallel worker to process.")
    
    args = parser.parse_args()
    
    run_shazam_preprocessing(args.image_folder, args.db_name, args.chunk_size)