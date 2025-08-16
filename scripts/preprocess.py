import argparse
import os
import sys
import numpy as np
from tqdm import tqdm
import time
from multiprocessing import Process, Queue

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.database import create_connection
# Only the GPU function is needed here now
from src.similarity import get_embeddings_batch
# Import the producer task
from src.data_loader import producer_task

def run_preprocessing(image_root_folder, db_name='images.db', batch_size=128):
    conn = create_connection(db_name)
    c = conn.cursor()

    c.execute("SELECT id, filepath FROM image_index WHERE is_processed = 0")
    unprocessed_images = c.fetchall()

    if not unprocessed_images:
        print("All images have already been processed. Nothing to do!")
        conn.close()
        return
    
    total_to_process = len(unprocessed_images)
    print(f"--- Found {total_to_process} images to process ---")

    # Create a queue to hold the processed results from the producer
    results_queue = Queue(maxsize=3)

    # Start the producer process in the background
    producer = Process(
        target=producer_task, 
        args=(image_root_folder, unprocessed_images, results_queue, batch_size)
    )
    producer.start()

    pbar = tqdm(total=total_to_process, desc="Processing Images")

    # --- Main Consumer Loop ---
    while True:
        # Get a batch of processed results from the queue
        processed_batch = results_queue.get()

        if processed_batch is None: # End of queue
            break

        process_start_time = time.time()
        
        # Separate successful results from failures
        successful_batch = []
        failed_ids = []
        for image_id, resized_rgb, color_vec, fft_hash in processed_batch:
            if resized_rgb is not None:
                successful_batch.append((image_id, resized_rgb, color_vec, fft_hash))
            else:
                pbar.write(f"[WARNING] File for ID {image_id} failed to process and was skipped.")
                failed_ids.append(image_id)
        
        if successful_batch:
            ids, resized_images_rgb, color_vecs, fft_hashes = zip(*successful_batch)
            
            gpu_start = time.time()
            embedding_batch = get_embeddings_batch(np.array(resized_images_rgb, dtype=np.float32))
            gpu_end = time.time()

            db_start = time.time()
            # Prepare data for bulk insert
            characteristics_data = []
            for i in range(len(ids)):
                characteristics_data.append((ids[i], color_vecs[i].tobytes(), embedding_batch[i].tobytes(), fft_hashes[i]))

            c.executemany("INSERT OR REPLACE INTO image_characteristics (image_id, feature_vector, embedding, fft_hash) VALUES (?, ?, ?, ?)", characteristics_data)
            successful_update_ids = [(1, image_id) for image_id in ids]
            c.executemany("UPDATE image_index SET is_processed = ? WHERE id = ?", successful_update_ids)
            db_end = time.time()

            total_process_time = time.time() - process_start_time
            pbar.set_postfix_str(
                f"GPU: {gpu_end - gpu_start:.2f}s | "
                f"DB: {db_end - db_start:.2f}s | "
                f"Total: {total_process_time:.2f}s/batch"
            )

        if failed_ids:
            failed_update_ids = [(1, image_id) for image_id in failed_ids]
            c.executemany("UPDATE image_index SET is_processed = ? WHERE id = ?", failed_update_ids)
        
        conn.commit()
        pbar.update(len(processed_batch))

    # Clean up the producer process
    producer.join()
    pbar.close()
    conn.close()
    print("\n--- Preprocessing Complete! ---")

if __name__ == '__main__':
    # --- IMPORTANT ---
    # This guard is essential for multiprocessing to work correctly on Windows
    parser = argparse.ArgumentParser(description="Process all unprocessed images using a prefetching pipeline.")
    parser.add_argument("image_folder", type=str, help="The CURRENT root folder of your image collection.")
    parser.add_argument("--db_name", type=str, default="images.db", help="The SQLite database file name.")
    parser.add_argument("--batch_size", type=int, default=256, help="Number of images to process in each batch.")
    
    args = parser.parse_args()
    
    run_preprocessing(args.image_folder, args.db_name, args.batch_size)
