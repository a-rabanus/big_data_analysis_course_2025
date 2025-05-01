import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity

class ImageProcessor:
    def __init__(self, directory):
        self.directory = directory
        self.images = []  # List of (filename, image) tuples
        self.downscaled_images = []
        
    def load_images(self):
        """Load all images from the specified directory using OpenCV"""
        self.images = []
        for filename in os.listdir(self.directory):
            filepath = os.path.join(self.directory, filename)
            if os.path.isfile(filepath):
                img = cv2.imread(filepath)
                if img is not None:
                    self.images.append((filename, img))
    
    def downscale_images(self, factor=0.01):
        """Downscale all loaded images by specified factor"""
        self.downscaled_images = []
        for filename, img in self.images:
            height, width = img.shape[:2]
            new_size = (int(width * factor), int(height * factor))
            downscaled_img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
            self.downscaled_images.append((filename, downscaled_img))


class DominantColorExtractor:
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.dominant_colors = []  # List of (filename, colors, percentages)
        self.normalized_colors = []  # List of (filename, normalized_colors, percentages)
    
    def extract_dominant_colors(self, downscaled_images):
        """Extract dominant colors from all downscaled images"""
        self.dominant_colors = []
        for filename, img in downscaled_images:
            # Convert color space (preserving original potentially flawed conversion)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            pixels = img.reshape(-1, 3)
            
            # Apply KMeans clustering
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
            kmeans.fit(pixels)
            
            # Process results
            dominant_colors = kmeans.cluster_centers_.astype(int)
            labels = kmeans.labels_
            cluster_sizes = np.bincount(labels)
            sorted_indices = np.argsort(cluster_sizes)[::-1]
            
            # Sort colors and percentages
            sorted_colors = dominant_colors[sorted_indices]
            sorted_percentages = (cluster_sizes[sorted_indices] / cluster_sizes.sum()) * 100
            
            self.dominant_colors.append((filename, sorted_colors, sorted_percentages))
        
        # Create normalized version
        self._create_normalized_colors()
    
    def _create_normalized_colors(self):
        """Create normalized color version (0-1 range)"""
        self.normalized_colors = []
        for filename, colors, percentages in self.dominant_colors:
            normalized = colors / 255
            self.normalized_colors.append((filename, normalized, percentages))


class SimilarityComparator:
    def __init__(self, normalized_colors, n_clusters):
        self.normalized_colors = normalized_colors
        self.n_clusters = n_clusters
    
    @staticmethod
    def euk_dis(img1_colors, img2_colors):
        """Calculate pairwise Euclidean distances between color sets"""
        distances = euclidean_distances(img1_colors, img2_colors)
        return np.diag(distances)
    
    @staticmethod
    def cos_dis(img1_colors, img2_colors):
        """Calculate pairwise cosine distances between color sets"""
        similarities = cosine_similarity(img1_colors, img2_colors)
        return 1 - np.diag(similarities)
    
    def weights_function(self, similarity_scores):
        """Apply weighting based on cluster order"""
        result = 0
        for i in range(len(similarity_scores)):
            result += similarity_scores[i] * (1 - (i/self.n_clusters))
        return result
    
    def compare_two_images(self, img1_colors, img2_colors):
        """Calculate weighted similarity between two images' color sets"""
        similarity_scores = self.euk_dis(img1_colors, img2_colors)
        return self.weights_function(similarity_scores)
    
    def find_similarities(self, target_idx):
        """Find top 5 similar images for target image"""
        target_file, target_colors, target_percentages = self.normalized_colors[target_idx]
        
        similarities = []
        for idx, (filename, colors, percentages) in enumerate(self.normalized_colors):
            if idx == target_idx:
                continue  # Skip reference image
            similarity = self.compare_two_images(target_colors, colors)
            similarities.append((filename, colors, percentages, similarity))
        
        # Sort by similarity (ascending - lower is more similar)
        return sorted(similarities, key=lambda x: x[3])[:5]


class ImageVisualizer:
    @staticmethod
    def display_image(images, img_id=None):
        """Display single image from images list"""
        if img_id is None:
            img_id = random.randint(0, len(images)-1)
        filename, img = images[img_id]
        plt.figure()
        plt.title(filename)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
    
    @staticmethod
    def plot_dominant_colors(dominant_colors, img_id=None):
        """Visualize dominant colors with percentages"""
        if img_id is None:
            img_id = random.randint(0, len(dominant_colors)-1)
        filename, colors, percentages = dominant_colors[img_id]
        
        plt.figure(figsize=(8, 2))
        plt.axis('off')
        plt.imshow([colors / 255])
        plt.title(filename)
        
        for i, (color, percentage) in enumerate(zip(colors, percentages)):
            plt.text(
                x=i/len(colors) + 0.17,
                y=0.5,
                s=f"{percentage:.1f}%",
                color="white" if np.mean(color) < 128 else "black",
                fontsize=12,
                ha="center",
                va="center",
                transform=plt.gca().transAxes
            )
        plt.show()
    
    @staticmethod
    def show_similar_images(original_images, similar_images_data):
        """Display similar images grid"""
        image_dict = {filename: img for filename, img in original_images}
        
        fig, axes = plt.subplots(1, 5, figsize=(20, 5))
        if len(similar_images_data) < 5:
            axes = axes.flat[:len(similar_images_data)]
        
        for ax, (filename, _, _, similarity) in zip(axes, similar_images_data[:5]):
            img = image_dict[filename]
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(f"{filename}\nSimilarity: {similarity:.2f}")
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()


# Main execution flow
N_CLUSTERS = 5
DIR = r"C:\Users\anton\OneDrive\Documents\HSD\sem4\DAISY_2025_images_for_bigdata"

# Initialize components
image_processor = ImageProcessor(DIR)
color_extractor = DominantColorExtractor(n_clusters=N_CLUSTERS)
visualizer = ImageVisualizer()

# Process images
image_processor.load_images()
image_processor.downscale_images(factor=0.01)

# Extract colors
color_extractor.extract_dominant_colors(image_processor.downscaled_images)

# Prepare comparator
similarity_comparator = SimilarityComparator(
    color_extractor.normalized_colors,
    n_clusters=N_CLUSTERS
)

# Find and display similar images
testing_id = -1  # Or use random.randint(0, len(image_processor.images)-1)
visualizer.display_image(image_processor.images, testing_id)
similar_images = similarity_comparator.find_similarities(testing_id)
visualizer.show_similar_images(image_processor.images, similar_images)