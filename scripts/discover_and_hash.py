import argparse
import os
import sys
import pathlib
import hashlib
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.database_setup import setup_database
from src.database import create_connection

def discover_and_hash_images(image_folder, db_name='images.db'):
    print("--- Starting discovery and hashing process ---")
    conn = create_connection(db_name)
    c = conn.cursor()
    
    print("Discovering all image files on disk...")
    all_disk_paths = []
    image_extensions = {".jpg", ".jpeg", ".png"}
    for dirpath, _, filenames in os.walk(image_folder):
        for filename in filenames:
            if pathlib.Path(filename).suffix.lower() in image_extensions:
                all_disk_paths.append(os.path.normpath(os.path.join(dirpath, filename)))
    print(f"Found {len(all_disk_paths)} total images.")

    images_to_log = []
    normalized_root = os.path.normpath(image_folder)
    print("Calculating unique hashes and relative paths for each image...")
    for path in tqdm(all_disk_paths, desc="Hashing Images"):
        # This is the key change: we calculate the relative path here.
        relative_path = os.path.relpath(path, normalized_root)
        
        # The hash is generated from the stable relative path.
        hash_id = hashlib.sha1(relative_path.encode('utf-8')).hexdigest()
        
        # We store the hash and the RELATIVE path in the database.
        images_to_log.append((hash_id, relative_path))

    print(f"Adding {len(images_to_log)} images to the database...")
    # The filepath column now receives the relative_path
    c.executemany("INSERT OR IGNORE INTO image_index (id, filepath) VALUES (?, ?)", images_to_log)
    conn.commit()
    conn.close()
    
    print("--- Discovery and hashing complete! ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Discover, hash, and log all images to the database.")
    parser.add_argument("image_folder", type=str, help="The root folder of your image collection.")
    parser.add_argument("--db_name", type=str, default="images.db", help="The SQLite database file name.")
    args = parser.parse_args()
    
    setup_database(args.db_name)
    discover_and_hash_images(args.image_folder, args.db_name)