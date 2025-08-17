import cv2
import matplotlib
import matplotlib.pyplot as plt
import os
import io
import base64
import numpy as np
from src.similarity import get_fft_data_for_analysis

# Use a non-interactive backend for Matplotlib, crucial for web servers
matplotlib.use('Agg')

def show_results(input_image_path, results, image_root_folder, top_n=5):
    """
    Displays the input image and the top N similar images for each category in a grid.
    """
    # Create a plot with 3 rows (one for each similarity metric)
    fig, axes = plt.subplots(3, top_n + 1, figsize=(20, 12))
    fig.suptitle('Image Similarity Search Results', fontsize=24)

    # --- Display the Input Image ---
    input_img = cv2.imread(input_image_path)
    input_img_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    
    # Place the input image in the first column of each row
    for i in range(3):
        axes[i, 0].imshow(input_img_rgb)
        axes[i, 0].set_title("Input Image")
        axes[i, 0].axis('off')

    # --- Display the Results for Each Category ---
    result_categories = {
        "Content (Embedding)": results['by_embedding'],
        "Color": results['by_color'],
        "Structure (FFT)": results['by_fft']
    }

    row_num = 0
    for category_title, result_df in result_categories.items():
        # Set the title for the row
        axes[row_num, 0].text(-0.2, 0.5, category_title, transform=axes[row_num, 0].transAxes, 
                               ha="right", va="center", fontsize=16, rotation=90)

        # Iterate through the top N matches in the DataFrame
        for i, (index, row) in enumerate(result_df.iterrows()):
            # Construct the full path to the result image
            full_path = os.path.join(image_root_folder, row['filepath'])
            
            try:
                result_img = cv2.imread(full_path)
                result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                
                # Display the image
                ax = axes[row_num, i + 1]
                ax.imshow(result_img_rgb)
                
                # Add the score/distance as the title
                score_text = ""
                if 'embedding_sim' in row:
                    score_text = f"Score: {row['embedding_sim']:.3f}"
                elif 'color_dist' in row:
                    score_text = f"Dist: {row['color_dist']:.0f}"
                elif 'fft_dist' in row:
                    score_text = f"Dist: {row['fft_dist']}"
                
                ax.set_title(score_text)
                ax.axis('off')
            except Exception as e:
                print(f"Warning: Could not load or display image {full_path}. Error: {e}")
                axes[row_num, i + 1].set_title("Load Error")
                axes[row_num, i + 1].axis('off')

        row_num += 1

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # ==============================================================================
# SECTION 1: COLOR PALETTE PLOT
# ==============================================================================
def plot_single_palette(ax, signature):
    # ... (This function is unchanged)
    sorted_signature = sorted(signature, key=lambda x: x[0], reverse=True)
    percentages = [item[0] for item in sorted_signature]
    colors = [item[-3:] / 255 for item in sorted_signature]
    left_edge = 0
    for i in range(len(percentages)):
        ax.barh(0, percentages[i], color=colors[i], edgecolor='black', height=0.5, left=left_edge)
        left_edge += percentages[i]
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

def create_color_palette_comparison_plot(query_features, match_features, query_title, match_title):
    # ... (This function is unchanged)
    fig, axes = plt.subplots(2, 1, figsize=(8, 2))
    axes[0].set_title(query_title, fontsize=10)
    plot_single_palette(axes[0], query_features['color'])
    axes[1].set_title(match_title, fontsize=10)
    plot_single_palette(axes[1], match_features['color'])
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return image_base64

# ==============================================================================
# SECTION 2: FFT SPECTRUM PLOT (NEW)
# ==============================================================================

def plot_single_fft(ax, magnitude_spectrum, rect_coords):
    """Helper function to plot a single FFT spectrum with its hash region and labels."""
    y_start, y_end, x_start, x_end = rect_coords
    
    # Use a logarithmic scale to make the high-frequency details visible
    log_spectrum = np.log(1 + magnitude_spectrum)
    ax.imshow(log_spectrum, cmap='gray')
    
    # --- THIS IS THE CHANGE ---
    # Add labels to the axes and a more descriptive title
    ax.set_title("Frequency Spectrum (Brightness = Magnitude)")
    ax.set_xlabel("Horizontal Frequency")
    ax.set_ylabel("Vertical Frequency")
    # Hide the tick values for a cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Draw a red rectangle around the area used for the hash
    rect = plt.Rectangle((x_start, y_start), x_end - x_start, y_end - y_start, 
                         edgecolor='red', facecolor='none', linewidth=2, label='Hash Region')
    ax.add_patch(rect)

def create_fft_comparison_plot(query_img, match_img, query_title, match_title):
    """
    Creates a plot comparing the FFT spectrums of two images.
    """
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    fig.suptitle('FFT Spectrum Comparison (Structural Similarity)', fontsize=16)

    # --- Process and plot query image ---
    query_spec, q_rect = get_fft_data_for_analysis(query_img)
    axes[0, 0].imshow(cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(query_title)
    axes[0, 0].axis('off')
    plot_single_fft(axes[1, 0], query_spec, q_rect)

    # --- Process and plot match image ---
    match_spec, m_rect = get_fft_data_for_analysis(match_img)
    axes[0, 1].imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(match_title)
    axes[0, 1].axis('off')
    plot_single_fft(axes[1, 1], match_spec, m_rect)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # --- (Buffer saving logic is unchanged) --- 
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return image_base64



# ==============================================================================
# SECTION 3: EMBEDDING HEATMAP PLOT (NEW)
# ==============================================================================
def create_embedding_comparison_plot(query_features, match_features, query_title, match_title, score):
    """
    Creates a plot comparing two embedding vectors as heatmaps.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 2))
    fig.suptitle(f'Embedding Comparison (Content Similarity)\nCosine Score: {score:.4f}', fontsize=16)

    # --- Plot Query Embedding Heatmap ---
    axes[0].imshow(query_features['embedding'].reshape(1, -1), cmap='viridis', aspect='auto')
    axes[0].set_title(query_title)
    axes[0].set_yticks([])
    
    # --- Plot Match Embedding Heatmap ---
    axes[1].imshow(match_features['embedding'].reshape(1, -1), cmap='viridis', aspect='auto')
    axes[1].set_title(match_title)
    axes[1].set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.85])
    
    # --- Save plot to in-memory buffer and return as base64 ---
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return image_base64
    