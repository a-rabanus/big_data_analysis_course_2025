import argparse
import os
import sys
import numpy as np
from tqdm import tqdm
import cv2
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.database import create_connection
from src.similarity import get_color_similarity_vector, get_fft_hash, get_embeddings_batch

def run_preprocessing(image_root_folder, db_name='images.db', batch_size=128):
    conn = create_connection(db_name)
    c = conn.cursor()

    # Find all images that have not been processed yet
    c.execute("SELECT id, filepath FROM image_index WHERE is_processed = 0")
    unprocessed_images = c.fetchall() # This now contains (id, relative_path)

    if not unprocessed_images:
        print("All images have already been processed. Nothing to do!")
        conn.close()
        return

    total_to_process = len(unprocessed_images)
    print(f"--- Found {total_to_process} images to process ---")
    pbar = tqdm(total=total_to_process, desc="Processing Images")

    # Process in batches
    for i in range(0, len(unprocessed_images), batch_size):
        batch_start_time = time.time()
        
        load_start = time.time()
        batch = unprocessed_images[i:i + batch_size]
        loaded_batch = []
        failed_ids = []
        for image_id, relative_path in batch:
            # --- KEY CHANGE: Reconstruct the full path ---
            # Joins the root folder you provide with the relative path from the DB.
            full_path = os.path.join(image_root_folder, relative_path)
            
            original_img = cv2.imread(full_path)
            if original_img is not None:
                resized_rgb = cv2.cvtColor(cv2.resize(original_img, (224, 224)), cv2.COLOR_BGR2RGB)
                loaded_batch.append((image_id, original_img, resized_rgb))
            else:
                pbar.write(f"[WARNING] Could not read image file, skipping: {full_path}")
                failed_ids.append(image_id)
        load_end = time.time()
        
        if loaded_batch:
            ids, original_images, resized_images_rgb = zip(*loaded_batch)
            
            gpu_start = time.time()
            embedding_batch = get_embeddings_batch(np.array(resized_images_rgb, dtype=np.float32))
            gpu_end = time.time()

            cpu_start = time.time()
            characteristics_data = []
            for j, img in enumerate(original_images):
                image_id = ids[j]
                color_vec = get_color_similarity_vector(img)
                fft_hash = get_fft_hash(img)
                embedding = embedding_batch[j]
                characteristics_data.append((image_id, color_vec.tobytes(), embedding.tobytes(), fft_hash))
            cpu_end = time.time()

            db_start = time.time()
            c.executemany("INSERT OR REPLACE INTO image_characteristics (image_id, feature_vector, embedding, fft_hash) VALUES (?, ?, ?, ?)", characteristics_data)
            successful_ids = [(1, image_id) for image_id in ids]
            c.executemany("UPDATE image_index SET is_processed = ? WHERE id = ?", successful_ids)
            db_end = time.time()
        
        if failed_ids:
            failed_update_ids = [(1, image_id) for image_id in failed_ids]
            c.executemany("UPDATE image_index SET is_processed = ? WHERE id = ?", failed_update_ids)

        conn.commit()
        
        pbar.update(len(batch))
        total_batch_time = time.time() - batch_start_time
        if loaded_batch:
            pbar.set_postfix_str(
                f"Load: {load_end - load_start:.2f}s | "
                f"GPU: {gpu_end - gpu_start:.2f}s | "
                f"CPU: {cpu_end - cpu_start:.2f}s | "
                f"DB: {db_end - db_start:.2f}s | "
                f"Total: {total_batch_time:.2f}s/batch"
            )

    pbar.close()
    conn.close()
    print("\n--- Preprocessing Complete! ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process all unprocessed images in the database.")
    # The image folder is now a required argument for this script as well.
    parser.add_argument("image_folder", type=str, help="The CURRENT root folder of your image collection.")
    parser.add_argument("--db_name", type=str, default="images.db", help="The SQLite database file name.")
    parser.add_argument("--batch_size", type=int, default=256, help="Number of images to process in each batch.")
    
    args = parser.parse_args()
    
    run_preprocessing(args.image_folder, args.db_name, args.batch_size)