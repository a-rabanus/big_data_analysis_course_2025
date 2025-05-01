# imports and path to images
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity

N_CLUSTERS = 5
DIR = r"C:\Users\anton\OneDrive\Documents\HSD\sem4\DAISY_2025_images_for_bigdata"

def load_images_from_directory(directory):
    """
    Load all images from the specified directory using OpenCV.
    """
    images = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            img = cv2.imread(filepath)
            if img is not None:
                images.append((filename, img))
    return images

def display_image(images, img_id=None):
    """
    Display a list of images using Matplotlib.
    """
    if img_id is None:
        img_id = random.randint(0, len(images) - 1)
    else:
        img_id = img_id
    filename, img = images[img_id]
    plt.figure()
    plt.title(filename)
    # Convert BGR to RGB for displaying with Matplotlib
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    pass

def downscale_image(image, factor=0.01):
    """
    Downscale images by a specified factor.
    """
    filename, img = image
    height, width = img.shape[:2]
    new_size = (int(width * factor), int(height * factor))
    downscaled_img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    return (filename, downscaled_img)

def extract_dominant_colors(image, n_clusters=3):
    """
    Extract dominant colors from an image using KMeans clustering.
    Returns a n_clusters long list of rgb values 
    """
    # Unpack the tuple to get the filename and image
    # Convert the image to RGB (from BGR)
    # Reshape the image to a 2D array of pixels where each row is a pixel and each column is a color channel

    filename, img = image
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    pixels = img.reshape(-1, 3)
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(pixels)
    
    # Get the cluster centers (dominant colors)
    dominant_colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    cluster_sizes = np.bincount(labels)

    # Get indices that would sort the cluster sizes in descending order
    sorted_indices = np.argsort(cluster_sizes)[::-1]

    # Sort both colors and cluster sizes using these indices
    dominant_colors = dominant_colors[sorted_indices]
    cluster_sizes = cluster_sizes[sorted_indices]
    
    # to percentages
    total_pixels = sum(cluster_sizes)
    percentages = (cluster_sizes / total_pixels) * 100

    return (filename, dominant_colors, percentages)

def plot_dominant_colors(all_colors, img_id=None):
    """
    Gets tuple with label and dominant colors and plots them
    """
    if img_id is None:
        img_id = random.randint(0, len(all_colors) - 1)
    else:
        img_id = img_id
        
    filename, colors, percentages = all_colors[img_id]
    
    # Create the plot
    plt.figure(figsize=(8, 2))
    plt.axis('off')
    plt.imshow([colors / 255])  # Normalize colors to [0, 1] for Matplotlib
    plt.title(filename)
    
    # Annotate with percentages
    for i, (color, percentage) in enumerate(zip(colors, percentages)):
        plt.text(
            x=i / len(colors) + 0.17,  # Position text slightly offset
            y=0.5,
            s=f"{percentage:.1f}%",
            color="white" if np.mean(color) < 128 else "black",  # Contrast text color
            fontsize=12,
            ha="center",
            va="center",
            transform=plt.gca().transAxes
        )
    
    plt.show()
    pass

def euk_dis(img_1, img_2):
    """
    Calculate the Euclidean distance between the colors. Smaller distances indicate higher similarity.

    dist(x, y) = sqrt(dot(x, x) - 2 * dot(x, y) + dot(y, y))

    [Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.euclidean_distances.html)
    
    this function outputs the distance from every entry to every other entry.
    We're only interested in the diagonal of this matrix


    input: img_1 (numpy array), img_2 (numpy array)
    
    output: float for every distande
    
    output interpretation: `0`: Identical colors, √3: Maximum possible distance for 3 dim
    """
    all_distances_between_images = euclidean_distances(img_1, img_2)
    distances = np.diag(all_distances_between_images)
    return distances.T

def cos_dis(img_1, img_2):
    """
    cosine_distance = 1 - cosine_similarity

    K(X, Y) = <X, Y> / (||X||*||Y||)

    [Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html)
    
    this function outputs the cosine similarity from every entry to every other entry.
    We're only interested in the diagonal of this matrix

    input: img_1 (numpy array), img_2 (numpy array)
    
    output: float
    
    output interpretation: 0 = identical, 2 = maximally dissimilar.
    """
    all_similarities_between_images = cosine_similarity(img_1, img_2)
    similarity = np.diag(all_similarities_between_images)
    return [1, 1, 1] - similarity

def weights_function(similarity_scores, n_clusters=N_CLUSTERS):
    result = 0
    for i in range(len(similarity_scores)):
        result += similarity_scores[i] * (1 - (i/n_clusters))
    return result 

def compare_two_images(img_1, img_2):
    similarity_scores = euk_dis(img_1, img_2)
    similarity = weights_function(similarity_scores)
    return similarity

def find_similarities(all_dominant_colors_normalized, img_idx_1):
    filename1, colors1, percentages1 = all_dominant_colors_normalized[img_idx_1]

    modified_list = []

    for idx, image_data in enumerate(all_dominant_colors_normalized):
        filename2, colors2, percentages2 = image_data
        
        if filename2 == filename1:
            # Add None similarity for the reference image
            modified_list.append((*image_data, None))
        else:
            # Calculate similarity and append with score
            similarity = compare_two_images(colors1, colors2)
            modified_list.append((*image_data, similarity))
    
    # Sort by similarity (position 3 in the tuple), exclude None values
    img_sorted_by_similarity = sorted(
        modified_list,
        key=lambda x: x[3] if x[3] is not None else float('inf')
    )
    
    # Return top 5 (excluding the reference image itself)
    return [item for item in img_sorted_by_similarity if item[3] is not None][:5]

def show_similar_images(images, list_of_images_to_show):
    """disclaimer: ai generated function!"""
    image_dict = {img[0]: img[1] for img in images}
    
    # Create a single figure with subplots
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))  # 1 row, 5 columns
    
    # If there are fewer than 5 images, adjust the axes array
    if len(list_of_images_to_show) < 5:
        axes = axes.flat[:len(list_of_images_to_show)]
    
    for ax, show_data in zip(axes, list_of_images_to_show[:5]):  # Only show first 5
        filename = show_data[0]
        if filename in image_dict:
            _, _, percentages, similarity = show_data
            original_img = image_dict[filename]
            
            # Plot on the current axis
            ax.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
            ax.set_title(f"{filename}\nSimilarity: {similarity:.2f}")
            ax.axis('off')
    plt.tight_layout()
    plt.show()


# Load and downscale images
images = load_images_from_directory(DIR)

# downscale images
downscaled_images = []
for img in images:
    downscaled_images.append(downscale_image(img))
# print(f"Downscaled {len(images)} images")

# extract dominant colors for all images
all_dominant_colors = []
for img in downscaled_images:
    all_dominant_colors.append(extract_dominant_colors(img, n_clusters=N_CLUSTERS))
# print(all_dominant_colors[testing_id][0])

# normalized
all_dominant_colors_normalized = []
for dominant_colors in all_dominant_colors:
    normalized_colors = dominant_colors[1] / 255
    norm_label = dominant_colors[0]
    percentages = dominant_colors[2]
    all_dominant_colors_normalized.append((norm_label, normalized_colors, percentages))

testing_id = -1#random.randint(0, len(images) - 1)

display_image(images, testing_id)
similar_images = find_similarities(all_dominant_colors_normalized, testing_id)

show_similar_images(images, similar_images)