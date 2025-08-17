from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import sys
import base64
import argparse
import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from engine import SearchEngine
# Import all our new plotting functions
from display import (
    create_color_palette_comparison_plot, 
    create_fft_comparison_plot, 
    create_embedding_comparison_plot
)

# --- (Argument parsing and Initialization is unchanged) ---
parser = argparse.ArgumentParser(description="Run the Flask Image Similarity Search App.")
parser.add_argument("image_folder", type=str, help="The absolute path to the root folder of your image dataset.")
args = parser.parse_args()
IMAGE_DATASET_ROOT = os.path.abspath(args.image_folder)
if not os.path.isdir(IMAGE_DATASET_ROOT):
    print(f"Error: The provided image folder does not exist: {IMAGE_DATASET_ROOT}")
    sys.exit(1)
print("--- Initializing Search Engine ---")
SEARCH_ENGINE = SearchEngine(db_name='images.db')
print("--- Search Engine Ready ---")

LAST_UPLOADED_FEATURES = None
LAST_UPLOADED_IMAGE_BYTES = None # Store the uploaded image bytes
app = Flask(__name__)

# --- (index, upload, search, serve_image routes are mostly unchanged) ---
@app.route('/')
def index(): return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global LAST_UPLOADED_FEATURES, LAST_UPLOADED_IMAGE_BYTES
    # ... (code is unchanged)
    if 'file' not in request.files: return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({"error": "No selected file"}), 400
    if file:
        image_bytes = file.read()
        LAST_UPLOADED_IMAGE_BYTES = image_bytes # Save for plotting
        try:
            LAST_UPLOADED_FEATURES = SEARCH_ENGINE.process_new_image(image_bytes)
        except Exception as e:
            return jsonify({"error": "Could not process image."}), 500
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        return jsonify({"message": "Image processed successfully", "image": image_base64})

@app.route('/search', methods=['POST'])
def search():
    # ... (code is unchanged)
    if LAST_UPLOADED_FEATURES is None: return jsonify({"error": "No image has been uploaded yet."}), 400
    results = SEARCH_ENGINE.find_similar(LAST_UPLOADED_FEATURES)
    json_results = {
        "embedding": results['by_embedding'][['filepath', 'embedding_sim']].to_dict(orient='records'),
        "color": results['by_color'][['filepath', 'color_dist']].to_dict(orient='records'),
        "fft": results['by_fft'][['filepath', 'fft_dist']].to_dict(orient='records'),
    }
    return jsonify(json_results)

@app.route('/images/<path:filepath>')
def serve_image(filepath):
    return send_from_directory(IMAGE_DATASET_ROOT, filepath)

# --- UPDATED: Analyze route now handles different plot types ---
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    if not data or 'filepath' not in data or 'type' not in data:
        return jsonify({"error": "Missing filepath or analysis type."}), 400
    
    analysis_type = data['type']
    match_filepath = data['filepath']
    
    if LAST_UPLOADED_FEATURES is None:
        return jsonify({"error": "Please upload an image first."}), 400
        
    match_features = SEARCH_ENGINE.get_features_by_filepath(match_filepath)
    if match_features is None:
        return jsonify({"error": "Could not find features for the selected image."}), 404

    plot_base64 = ""
    query_title = "Uploaded Image"
    match_title = f"Match: {os.path.basename(match_filepath)}"

    if analysis_type == 'color':
        plot_base64 = create_color_palette_comparison_plot(
            LAST_UPLOADED_FEATURES, match_features, query_title, match_title
        )
    elif analysis_type == 'fft':
        # For FFT, we need to load the actual image files
        query_img_nparr = np.frombuffer(LAST_UPLOADED_IMAGE_BYTES, np.uint8)
        query_img = cv2.imdecode(query_img_nparr, cv2.IMREAD_COLOR)
        match_img_path = os.path.join(IMAGE_DATASET_ROOT, match_filepath)
        match_img = cv2.imread(match_img_path)
        plot_base64 = create_fft_comparison_plot(query_img, match_img, query_title, match_title)
        
    elif analysis_type == 'embedding':
        # We need the similarity score for the title
        match_row = SEARCH_ENGINE.features_df[SEARCH_ENGINE.features_df['filepath'] == match_filepath]
        score = match_row.iloc[0]['embedding_sim']
        plot_base64 = create_embedding_comparison_plot(
            LAST_UPLOADED_FEATURES, match_features, query_title, match_title, score
        )
    
    return jsonify({"plot": plot_base64})

if __name__ == '__main__':
    print(f"Serving images from: {IMAGE_DATASET_ROOT}")
    app.run(host='0.0.0.0', port=5000, debug=False) # Turn off debug mode for cleaner logs