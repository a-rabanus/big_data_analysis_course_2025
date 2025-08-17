import cv2
import numpy as np
import pandas as pd
import time
from src.similarity import (
    get_color_similarity_vector, 
    get_fft_hash, 
    get_embeddings_batch,
    calculate_color_distance,
    calculate_fft_hash_distance,
    calculate_embedding_similarity  # We still use this, but not in a loop
)
from src.database import load_all_features_to_dataframe
# --- NEW IMPORT ---
# We'll use the main cosine_similarity function for the vectorized calculation
from sklearn.metrics.pairwise import cosine_similarity


class SearchEngine:
    def __init__(self, db_name='images.db'):
        """
        Initializes the search engine by loading all features from the database.
        """
        start_time = time.time()
        self.features_df = load_all_features_to_dataframe(db_name)
        end_time = time.time()
        print(f"[DEBUG] Database loading took {end_time - start_time:.2f} seconds.")
        
        # --- NEW: Pre-stack embeddings for performance ---
        # Stack all embedding vectors into a single large NumPy matrix for fast computation.
        # This is a one-time cost at startup.
        print("[DEBUG] Pre-stacking embeddings for vectorized search...")
        self.all_embeddings = np.vstack(self.features_df['embedding'].values)
        self.all_fft_hashes = np.vstack(self.features_df['fft_hash_array'].values)
        
        print("[DEBUG] Stacking complete.")


    def process_new_image(self, image_bytes):
        """
        Loads a new image from its byte data and calculates all feature vectors.
        """
        print(f"\n[DEBUG] Processing new uploaded image...")
        start_time = time.time()
        
        # Convert the byte stream to a NumPy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        # Decode the array into an image
        original_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if original_img is None:
            raise ValueError("Could not decode the uploaded image file.")
        
        color_vec = get_color_similarity_vector(original_img)
        fft_hash = get_fft_hash(original_img)
        
        resized_rgb = cv2.cvtColor(cv2.resize(original_img, (224, 224)), cv2.COLOR_BGR2RGB)
        embedding = get_embeddings_batch(np.array([resized_rgb], dtype=np.float32))[0]
        
        end_time = time.time()
        print(f"[DEBUG] New image processing took {end_time - start_time:.2f} seconds.")
        
        return {"color": color_vec, "fft": fft_hash, "embedding": embedding}

    def find_similar(self, new_image_features, top_n=5):
        print("\n[DEBUG] Finding similar images...")
        total_start_time = time.time()
        
        # --- 1. Vectorized Embedding Similarity ---
        start_time = time.time()
        new_embedding = new_image_features['embedding'].reshape(1, -1)
        sim_scores = cosine_similarity(new_embedding, self.all_embeddings)[0]
        self.features_df['embedding_sim'] = sim_scores
        top_embedding = self.features_df.sort_values(by='embedding_sim', ascending=False).head(top_n)
        end_time = time.time()
        print(f"[DEBUG]   - Embedding search took {end_time - start_time:.4f} seconds.")

        # --- 2. Color Similarity (still uses .apply) ---
        start_time = time.time()
        self.features_df['color_dist'] = self.features_df['feature_vector'].apply(
            lambda x: calculate_color_distance(new_image_features['color'], x)
        )
        top_color = self.features_df.sort_values(by='color_dist', ascending=True).head(top_n)
        end_time = time.time()
        print(f"[DEBUG]   - Color search took {end_time - start_time:.4f} seconds.")

        # --- 3. OPTIMIZED: Vectorized FFT Hash Similarity ---
        start_time = time.time()
        # Convert the new image's hash string to a NumPy array of integers
        new_fft_hash_array = np.array(list(new_image_features['fft']), dtype=np.int8)
        
        # Use NumPy broadcasting to find the number of differing bits for all images at once.
        # np.sum(..., axis=1) calculates the Hamming distance.
        fft_distances = np.sum(self.all_fft_hashes != new_fft_hash_array, axis=1)
        
        self.features_df['fft_dist'] = fft_distances
        top_fft = self.features_df.sort_values(by='fft_dist', ascending=True).head(top_n)
        end_time = time.time()
        print(f"[DEBUG]   - FFT search took {end_time - start_time:.4f} seconds.")
        
        total_end_time = time.time()
        print(f"[DEBUG] Total search time: {total_end_time - total_start_time:.2f} seconds.")
        
        return {"by_embedding": top_embedding, "by_color": top_color, "by_fft": top_fft}
    
    def get_features_by_filepath(self, filepath):
        """Retrieves the feature vectors for a given relative filepath."""
        # Find the row in the DataFrame that matches the filepath
        image_data = self.features_df[self.features_df['filepath'] == filepath]
        if image_data.empty:
            return None
        # Return the first match's features as a dictionary
        return {
            "color": image_data.iloc[0]['feature_vector'],
            "fft": image_data.iloc[0]['fft_hash'],
            "embedding": image_data.iloc[0]['embedding']
        }