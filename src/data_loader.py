import cv2
import numpy as np
import os
from multiprocessing import Pool, cpu_count
# The worker needs access to the CPU-bound similarity functions
from src.similarity import get_color_similarity_vector, get_fft_hash

# The worker's job is now to load AND perform CPU calculations.
def worker_task(path_tuple):
    """
    Loads an image, calculates CPU-bound features, resizes for the GPU,
    and returns only the processed results.
    """
    image_id, path = path_tuple
    try:
        original_img = cv2.imread(path)
        if original_img is None:
            return (image_id, None, None, None) # Return None for all features

        # --- Perform CPU-bound work inside the worker ---
        color_vec = get_color_similarity_vector(original_img)
        fft_hash = get_fft_hash(original_img)

        # --- Prepare the smaller resized image for the GPU ---
        resized_rgb = cv2.cvtColor(cv2.resize(original_img, (224, 224)), cv2.COLOR_BGR2RGB)
        
        # Return the ID and the small, processed data. The large original_img is discarded.
        return (image_id, resized_rgb, color_vec, fft_hash)
    except Exception:
        return (image_id, None, None, None)

def producer_task(image_root_folder, unprocessed_list, queue, batch_size):
    """
    The main function for the producer process. It creates its own worker pool
    to load images and perform CPU-bound preprocessing in parallel.
    """
    num_workers = max(1, cpu_count() - 4)
    
    with Pool(processes=num_workers) as pool:
        for i in range(0, len(unprocessed_list), batch_size):
            batch_to_load = unprocessed_list[i:i + batch_size]
            paths_to_load = [(img_id, os.path.join(image_root_folder, rel_path)) for img_id, rel_path in batch_to_load]
            
            # Use the pool to run the full worker_task on each image path
            processed_batch_results = pool.map(worker_task, paths_to_load)
            
            # Put the processed results onto the queue
            queue.put(processed_batch_results)

    # Signal that the work is done
    queue.put(None)
