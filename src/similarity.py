import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torchvision import models, transforms

# --- Color and FFT Similarity Functions ---

def get_color_similarity_vector(image, clusters=5):
    image_resized = cv2.resize(image, (100, 100))
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    pixels = image_rgb.reshape(-1, 3)
    kmeans = KMeans(n_clusters=clusters, n_init='auto', random_state=42)
    kmeans.fit(pixels)
    counts = np.bincount(kmeans.labels_) / len(kmeans.labels_)
    centers = kmeans.cluster_centers_
    signature = np.hstack([counts.reshape(-1, 1), centers]).astype(np.float32)
    return signature

def get_fft_hash(image):
    image_resized = cv2.resize(image, (32, 32))
    gray_image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    fft_transform = np.fft.fft2(gray_image)
    fft_shift = np.fft.fftshift(fft_transform)
    magnitude_spectrum = np.abs(fft_shift)
    hash_array = magnitude_spectrum[10:22, 10:22].flatten()
    median_val = np.median(hash_array)
    hash_string = "".join(['1' if i > median_val else '0' for i in hash_array])
    return hash_string

# --- PyTorch-Based Embedding Calculation (Restructured for Stability) ---
_model, _device, _preprocess = None, None, None

def _initialize_model():
    """Initializes the model, device, and transforms ONCE."""
    global _model, _device, _preprocess
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

def get_embeddings_batch(image_batch_numpy):
    """Generates embeddings for a batch, initializing the model on the first call."""
    if _model is None:
        _initialize_model()
    image_tensors = torch.stack([_preprocess(img) for img in image_batch_numpy])
    image_tensors = image_tensors.to(_device)
    with torch.no_grad():
        embeddings = _model(image_tensors)
    return embeddings.cpu().numpy()