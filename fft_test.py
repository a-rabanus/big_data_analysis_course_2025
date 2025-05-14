import os
import cv2
import numpy as np
from scipy.fft import fft2, fftshift
from skimage.measure import ransac
from skimage.transform import AffineTransform
from collections import defaultdict
from math import atan2, hypot, pi
import gc

# Image Loading Functions
def load_images_from_directory(directory):
    """Load all images from directory with error handling"""
    images = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            try:
                img = cv2.imread(filepath)
                if img is not None:
                    images.append((filename, img))
                else:
                    print(f"Warning: Could not read {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
    return images

def downscale_image(image, factor=0.25):
    """Safe downscaling with size validation"""
    filename, img = image
    if img.size == 0:
        return (filename, np.array([]))
    
    h, w = img.shape[:2]
    new_w = max(1, int(w * factor))
    new_h = max(1, int(h * factor))
    return (filename, cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA))

# Core Algorithm Implementation
def detect_spectral_peaks(img, top_n=100, threshold=0.15):
    """Memory-efficient peak detection"""
    if img.size == 0 or len(img.shape) != 3:
        return []
    
    try:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    except:
        return []
    
    peaks = []
    for i in range(3):
        channel = lab[:, :, i].astype(np.float32)
        rows, cols = channel.shape
        
        # Windowing
        win = np.outer(np.hamming(rows), np.hamming(cols))
        fft = fftshift(fft2(channel * win))
        mag = np.log1p(np.abs(fft))
        
        # Peak detection
        y, x = np.indices(mag.shape)
        local_max = (mag > threshold * mag.max()) & \
                    (mag == cv2.dilate(mag, np.ones((3,3))))
        peaks.extend([(y, x, mag[y,x]) for y, x in zip(*np.where(local_max))])
    
    return sorted(peaks, key=lambda x: -x[2])[:top_n]

def form_anchor_pairs(peaks, max_dist=50, max_pairs=200):
    """Controlled pair generation"""
    pairs = []
    count = 0
    for i in range(len(peaks)):
        for j in range(i+1, len(peaks)):
            if count >= max_pairs:
                return pairs
            y1, x1, m1 = peaks[i]
            y2, x2, m2 = peaks[j]
            dx, dy = x2 - x1, y2 - y1
            if dx**2 + dy**2 <= max_dist**2:
                pairs.append((m1, m2, dx, dy, abs(m1 - m2)))
                count += 1
    return pairs

def create_visual_hash(pair, angle_bins=8):
    """Safe hashing with input validation"""
    try:
        m1, m2, dx, dy, _ = pair
        r = int(hypot(dx, dy))
        theta = int((atan2(dy, dx) + pi) * angle_bins / (2*pi)) % angle_bins
        return (r, theta, int(m1*100), int(m2*100))
    except:
        return (0, 0, 0, 0)

# Database Class
class VisualHashDB:
    def __init__(self):
        self.index = defaultdict(list)
        self.metadata = {}
    
    def add_image(self, img_id, image):
        if image.size == 0:
            return
            
        try:
            rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            return
            
        peaks = detect_spectral_peaks(rgb_img)
        pairs = form_anchor_pairs(peaks)
        
        self.metadata[img_id] = {
            'dim': rgb_img.shape,
            'peaks': len(peaks),
            'pairs': len(pairs)
        }
        
        for pair in pairs:
            h = create_visual_hash(pair)
            if sum(h) > 0:  # Skip invalid hashes
                self.index[h].append((img_id, pair[2], pair[3]))
        
        gc.collect()
    
    def query_image(self, query_img, min_matches=15):
        if query_img.size == 0:
            return None, 0
            
        try:
            rgb_query = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        except:
            return None, 0
            
        peaks = detect_spectral_peaks(rgb_query)
        pairs = form_anchor_pairs(peaks)
        
        candidates = defaultdict(list)
        for pair in pairs:
            h = create_visual_hash(pair)
            for entry in self.index.get(h, []):
                candidates[entry[0]].append((
                    (pair[2], pair[3]),
                    (entry[1], entry[2])
                ))
        
        # RANSAC verification
        for img_id, matches in candidates.items():
            if len(matches) >= min_matches:
                src = np.array([m[0] for m in matches])
                dst = np.array([m[1] for m in matches])
                
                try:
                    model, inliers = ransac(
                        (src, dst),
                        AffineTransform,
                        min_samples=3,
                        residual_threshold=2.0,
                        max_trials=100
                    )
                    if np.sum(inliers) >= min_matches:
                        return img_id, np.sum(inliers)
                except:
                    continue
        
        return None, 0

# Main Execution
if __name__ == "__main__":
    # Configuration
    DIR = r"C:\Users\anton\OneDrive\Documents\HSD\sem4\DAISY_2025_images_for_bigdata"
    QUERY_DIR = os.path.join(DIR, "queries")
    DOWNSCALE_FACTOR = 0.25
    
    # Initialize database
    db = VisualHashDB()
    
    # Load and process dataset
    print("Loading dataset...")
    dataset_images = load_images_from_directory(DIR)
    print(f"Found {len(dataset_images)} dataset images")
    
    for filename, img in dataset_images:
        _, downscaled = downscale_image((filename, img), DOWNSCALE_FACTOR)
        if downscaled.size > 0:
            db.add_image(filename, downscaled)
    
    # Load and process queries
    print("\nLoading queries...")
    query_images = load_images_from_directory(QUERY_DIR)
    print(f"Found {len(query_images)} query images")
    
    for filename, img in query_images:
        _, downscaled = downscale_image((filename, img), DOWNSCALE_FACTOR)
        if downscaled.size == 0:
            continue
            
        print(f"\nQuerying: {filename}")
        match_id, confidence = db.query_image(downscaled)
        
        if match_id:
            print(f"Match found: {match_id} (confidence: {confidence})")
        else:
            print("No match found")
    
    print("\nProcessing complete")