import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torchvision import models, transforms
import time
from scipy.ndimage import maximum_filter
from collections import Counter

# --- Color and FFT Similarity Functions ---

def get_color_similarity_vector(image, clusters=5):
    """
    Calculates a color signature based on both the color values (RGB)
    and the spatial position (x, y) of the pixels.
    """
    image_resized = cv2.resize(image, (100, 100))
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    height, width, _ = image_rgb.shape
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    pixels = image_rgb.reshape(-1, 3)
    coords = np.stack([y_coords.flatten(), x_coords.flatten()], axis=1)
    data = np.hstack([coords * (255 / 100), pixels])
    kmeans = KMeans(n_clusters=clusters, n_init='auto', random_state=42)
    kmeans.fit(data)
    counts = np.bincount(kmeans.labels_) / len(kmeans.labels_)
    centers = kmeans.cluster_centers_
    signature = np.hstack([counts.reshape(-1, 1), centers]).astype(np.float32)
    return signature

def get_fft_hash(image):
    """
    This is for PREPROCESSING. It returns ONLY the hash string for the database.
    """
    image_resized = cv2.resize(image, (32, 32))
    gray_image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    fft_transform = np.fft.fft2(gray_image)
    fft_shift = np.fft.fftshift(fft_transform)
    magnitude_spectrum = np.abs(fft_shift)
    
    y_start, y_end = 2, 30
    x_start, x_end = 2, 30
    hash_region = magnitude_spectrum[y_start:y_end, x_start:x_end]
    
    hash_array = hash_region.flatten()
    median_val = np.median(hash_array)
    hash_string = "".join(['1' if i > median_val else '0' for i in hash_array])
    
    return hash_string # Only returns the string

def get_fft_data_for_analysis(image):
    """
    This is for ANALYSIS. It returns all data needed for plotting.
    """
    image_resized = cv2.resize(image, (32, 32))
    gray_image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    fft_transform = np.fft.fft2(gray_image)
    fft_shift = np.fft.fftshift(fft_transform)
    magnitude_spectrum = np.abs(fft_shift)
    
    y_start, y_end = 2, 30
    x_start, x_end = 2, 30
    
    # Return everything needed for plotting
    return magnitude_spectrum, (y_start, y_end, x_start, x_end)

# --- PyTorch-Based Embedding Calculation (Restructured for Stability) ---
_model, _device, _preprocess = None, None, None

def _initialize_model():
    """Initializes the model, device, and transforms ONCE."""
    global _model, _device, _preprocess
    print("\n[DEBUG] Initializing PyTorch Model and moving to GPU...")
    start_time = time.time()
    print("\n--- Initializing PyTorch Model for GPU ---")
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {_device}")
    _model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    _model.classifier = torch.nn.Identity()
    _model.to(_device)
    _model.eval()
    _preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    end_time = time.time()
    print(f"[DEBUG] Model initialization complete. (Took {end_time - start_time:.2f} seconds)")


def get_embeddings_batch(image_batch_numpy):
    """Generates embeddings for a batch, initializing the model on the first call."""
    if _model is None:
        _initialize_model()
    image_tensors = torch.stack([_preprocess(img) for img in image_batch_numpy])
    image_tensors = image_tensors.to(_device)
    with torch.no_grad():
        embeddings = _model(image_tensors)
    return embeddings.cpu().numpy()

def calculate_embedding_similarity(emb1, emb2):
    """
    Calculates the cosine similarity between two embeddings.
    Higher is better (max is 1.0).
    """
    # Reshape to 2D arrays as expected by the function
    return cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]

def calculate_color_distance(sig1, sig2):
    """
    Calculates the Earth Mover's Distance between two color signatures.
    Lower is better (min is 0.0).
    """
    # cv2.EMD requires signatures to be of type float32
    return cv2.EMD(sig1, sig2, cv2.DIST_L2)[0]

def calculate_fft_hash_distance(hash1, hash2):
    """
    Calculates the Hamming distance between two FFT hashes.
    Lower is better (min is 0).
    """
    # This measures how many bits are different between the two hash strings.
    return np.sum([c1 != c2 for c1, c2 in zip(hash1, hash2)])
