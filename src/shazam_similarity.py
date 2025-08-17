import cv2
import numpy as np
from scipy.ndimage import maximum_filter
from collections import Counter
import numba # Import the JIT compiler

# ==============================================================================
# SHAZAM-STYLE FFT FINGERPRINTING (WITH JIT COMPILATION)
# ==============================================================================

# --- Configuration Constants ---
PEAK_BOX_SIZE = 15
TARGET_ZONE_ANCHORS = 5
TARGET_ZONE_WIDTH = 100
TARGET_ZONE_HEIGHT = 100

@numba.jit(nopython=True)
def _create_fingerprints_jit(sorted_keypoints):
    """
    A JIT-compiled helper function to handle the core fingerprinting loop.
    """
    fingerprints = set()
    num_keypoints = len(sorted_keypoints)

    for i in range(num_keypoints):
        anchor_point = sorted_keypoints[i]
        y_min, y_max = anchor_point[0], anchor_point[0] + TARGET_ZONE_HEIGHT
        x_min, x_max = anchor_point[1], anchor_point[1] + TARGET_ZONE_WIDTH
        
        neighbors = []
        for j in range(i + 1, num_keypoints):
            p = sorted_keypoints[j]
            if y_min <= p[0] < y_max and x_min <= p[1] < x_max:
                neighbors.append(p)
        
        if neighbors:
            # --- THIS IS THE FIX ---
            # Instead of using a lambda function for sorting, we pre-calculate
            # the distances in a simple loop that Numba can easily compile.
            
            # Create an array to hold the distances
            distances = np.empty(len(neighbors), dtype=np.float64)
            for k in range(len(neighbors)):
                p = neighbors[k]
                dist_sq = (p[0] - anchor_point[0])**2 + (p[1] - anchor_point[1])**2
                distances[k] = dist_sq
            
            # Get the indices that would sort the distances array
            sorted_indices = np.argsort(distances)
            
            # Take the closest N neighbors based on the sorted indices
            num_to_pair = min(len(neighbors), TARGET_ZONE_ANCHORS)
            for k in range(num_to_pair):
                neighbor_point = neighbors[sorted_indices[k]]
                dy = neighbor_point[0] - anchor_point[0]
                dx = neighbor_point[1] - anchor_point[1]
                
                hash_tuple = (dy, dx)
                anchor_tuple = (anchor_point[0], anchor_point[1])
                fingerprints.add((hash_tuple, anchor_tuple))
                
    return fingerprints

def get_shazam_fingerprints(image):
    """
    Generates a set of robust fingerprints for an image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_small = cv2.resize(gray, (512, 512))

    harris_corners = cv2.cornerHarris(img_small, blockSize=2, ksize=3, k=0.04)
    harris_corners = cv2.dilate(harris_corners, None)
    local_max = maximum_filter(harris_corners, size=PEAK_BOX_SIZE)
    keypoints_mask = (harris_corners == local_max) & (harris_corners > 0.01 * harris_corners.max())
    keypoint_coords = np.argwhere(keypoints_mask)
    
    # Convert to a NumPy array for Numba
    sorted_keypoints = keypoint_coords[np.argsort(keypoint_coords[:, 0])]
    
    fingerprints_jit = _create_fingerprints_jit(sorted_keypoints)

    # Convert back to the desired format (string hash)
    fingerprints = set()
    for (dy, dx), (ay, ax) in fingerprints_jit:
        hash_val = f"{dy}|{dx}"
        fingerprints.add((hash_val, (ay, ax)))
        
    return fingerprints, keypoint_coords

@numba.jit(nopython=True)
def _calculate_offsets_jit(matches):
    """JIT-compiled helper to calculate offsets between matching keypoints."""
    offsets = np.empty((len(matches), 2), dtype=np.int32)
    for i in range(len(matches)):
        p2_y, p2_x = matches[i, 2], matches[i, 3]
        p1_y, p1_x = matches[i, 0], matches[i, 1]
        offsets[i, 0] = p2_y - p1_y
        offsets[i, 1] = p2_x - p1_x
    return offsets

def calculate_shazam_similarity_fast(query_fingerprints, db_matches):
    """
    Calculates similarity using a JIT-compatible offset counting method.
    """
    query_map = {}
    for h, p in query_fingerprints:
        if h not in query_map: query_map[h] = []
        query_map[h].append(p)

    matches = []
    for db_hash, candidates in db_matches.items():
        if db_hash in query_map:
            for query_point in query_map[db_hash]:
                for image_id, db_y, db_x in candidates:
                    matches.append((query_point[0], query_point[1], db_y, db_x, image_id))

    if not matches: return []

    match_array = np.array(matches, dtype=[('qy', 'i4'), ('qx', 'i4'), ('dby', 'i4'), ('dbx', 'i4'), ('id', 'U40')])
    offsets = _calculate_offsets_jit(match_array[['qy', 'qx', 'dby', 'dbx']].view(np.int32).reshape(-1, 4))
    
    offset_tuples = [tuple(row) for row in offsets]
    unique_offsets, counts = np.unique(offset_tuples, axis=0, return_counts=True)
    
    if len(counts) == 0: return []
    
    best_offset_index = np.argmax(counts)
    best_offset = unique_offsets[best_offset_index]
    
    consistent_mask = (offsets[:, 0] == best_offset[0]) & (offsets[:, 1] == best_offset[1])
    consistent_matches = match_array[consistent_mask]

    image_ids, counts = np.unique(consistent_matches['id'], return_counts=True)
    top_matches = sorted(zip(image_ids, counts), key=lambda item: item[1], reverse=True)
    
    return top_matches[:5]