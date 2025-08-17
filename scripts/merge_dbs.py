import sqlite3
import glob
import os
from tqdm import tqdm
import argparse

def merge_temp_databases(main_db_path, temp_files_folder):
    """
    Merges all temporary Shazam fingerprint databases into the main database.
    """
    # --- Find all the temporary database files ---
    temp_db_files = glob.glob(os.path.join(temp_files_folder, '_temp_chunk_*.db'))
    
    if not temp_db_files:
        print("No temporary database files ('_temp_chunk_*.db') found to merge.")
        return

    print(f"Found {len(temp_db_files)} temporary database files to merge.")

    # --- Connect to the main database ---
    main_conn = sqlite3.connect(main_db_path)
    main_c = main_conn.cursor()

    print("Clearing any existing data in the 'shazam_fingerprints' table...")
    main_c.execute("DELETE FROM shazam_fingerprints")
    main_conn.commit()

    # --- REDUCE PHASE: Merge temporary databases ---
    print("\nStarting the merge process...")
    
    # Use a transaction for a massive speedup on insertion
    main_c.execute("BEGIN TRANSACTION;")
    
    total_fingerprints = 0
    for temp_db_path in tqdm(temp_db_files, desc="Merging Chunks"):
        try:
            temp_conn = sqlite3.connect(temp_db_path)
            temp_c = temp_conn.cursor()
            
            # Read all fingerprints from the temp file
            temp_c.execute("SELECT hash, image_id, anchor_y, anchor_x FROM shazam_fingerprints")
            fingerprints_from_chunk = temp_c.fetchall()
            
            if fingerprints_from_chunk:
                # Insert them into the main database
                main_c.executemany("INSERT INTO shazam_fingerprints (hash, image_id, anchor_y, anchor_x) VALUES (?, ?, ?, ?)", fingerprints_from_chunk)
                total_fingerprints += len(fingerprints_from_chunk)
            
            temp_conn.close()
        except Exception as e:
            print(f"\nWarning: Could not process file {temp_db_path}. Error: {e}")

    # Commit all the merged data at once
    print("\nCommitting all changes to the main database (this may take a moment)...")
    main_c.execute("COMMIT;")
    main_conn.close()
    
    print(f"\n--- Merge Complete! ---")
    print(f"Successfully inserted a total of {total_fingerprints} fingerprints into '{os.path.basename(main_db_path)}'.")

    # --- CLEANUP PHASE ---
    cleanup = input("\nDo you want to delete the temporary chunk files now? (y/n): ")
    if cleanup.lower() == 'y':
        print("Cleaning up temporary files...")
        for temp_db in temp_db_files:
            os.remove(temp_db)
        print("Cleanup complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merge temporary Shazam fingerprint databases into the main database.")
    parser.add_argument("main_db_path", type=str, help="The full path to your main 'images.db' file (e.g., 'I:\\path\\to\\images.db').")
    parser.add_argument("temp_files_folder", type=str, help="The path to the folder containing the '_temp_chunk_*.db' files.")
    
    args = parser.parse_args()
    
    merge_temp_databases(args.main_db_path, args.temp_files_folder)