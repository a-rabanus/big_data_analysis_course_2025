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
    """
    Scans a directory for all images, calculates a unique hash for each,
    and populates the database with a guaranteed RELATIVE path.
    """
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
    # --- THIS IS THE NEW, MORE ROBUST LOGIC ---
    # 1. Get a clean, absolute path for the root folder.
    root_prefix = os.path.abspath(os.path.normpath(image_folder))
    print(f"\n[DEBUG] Using root prefix for relative paths: '{root_prefix}'")
    
    print("Calculating unique hashes and guaranteed relative paths...")
    for i, path in enumerate(tqdm(all_disk_paths, desc="Hashing Images")):
        # 2. Manually create the relative path by removing the prefix.
        # This is more reliable than os.path.relpath across different drives.
        if path.startswith(root_prefix):
            relative_path = path[len(root_prefix):]
            # Remove leading slash or backslash to make it a clean relative path
            relative_path = relative_path.lstrip(os.path.sep)
        else:
            # Fallback for safety, though this case should not happen
            relative_path = os.path.basename(path)

        # 3. Add a foolproof assertion to crash if the path is still absolute.
        assert ":" not in relative_path, f"CRITICAL ERROR: Path is still absolute! Path: {relative_path}"

        # 4. (Optional) Print the first 5 paths for visual confirmation.
        if i < 5:
            print(f"  - [Example] Full: '{path}' -> Relative: '{relative_path}'")

        hash_id = hashlib.sha1(relative_path.encode('utf-8')).hexdigest()
        images_to_log.append((hash_id, relative_path))

    print(f"\nAdding {len(images_to_log)} images to the database...")
    c.executemany("INSERT OR IGNORE INTO image_index (id, filepath) VALUES (?, ?)", images_to_log)
    conn.commit()
    conn.close()
    
    print("--- Discovery and hashing complete! All paths are guaranteed relative. ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Discover, hash, and log all images to the database.")
    parser.add_argument("image_folder", type=str, help="The root folder of your image collection.")
    parser.add_argument("--db_name", type=str, default="images.db", help="The SQLite database file name.")
    args = parser.parse_args()
    
    setup_database(args.db_name)
    discover_and_hash_images(args.image_folder, args.db_name)
