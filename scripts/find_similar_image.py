import argparse
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.engine import SearchEngine
from src.display import show_results # Import the new display function

def main(image_path, image_root_folder, top_n=5, db_name='images.db', no_display=False):
    """
    Main function to run the image similarity search from the command line.
    """
    script_start_time = time.time()
    
    engine = SearchEngine(db_name)

    try:
        new_features = engine.process_new_image(image_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    results = engine.find_similar(new_features, top_n)

    # --- Display Text Results (Always) ---
    print("\n--- Top 5 Matches by Content (Embedding Similarity) ---")
    for i, row in results['by_embedding'].reset_index(drop=True).iterrows():
        print(f"  - Match {i+1}: {row['filepath']} (Score: {row['embedding_sim']:.4f})")

    print("\n--- Top 5 Matches by Color (Earth Mover's Distance) ---")
    for i, row in results['by_color'].reset_index(drop=True).iterrows():
        print(f"  - Match {i+1}: {row['filepath']} (Distance: {row['color_dist']:.2f})")

    print("\n--- Top 5 Matches by Structure (FFT Hash Distance) ---")
    for i, row in results['by_fft'].reset_index(drop=True).iterrows():
        print(f"  - Match {i+1}: {row['filepath']} (Distance: {row['fft_dist']})")

    script_end_time = time.time()
    print(f"\n[DEBUG] Total script execution time: {script_end_time - script_start_time:.2f} seconds.")

    # --- Display Visual Results (Optional) ---
    if not no_display:
        print("\nDisplaying visual results... (Close the plot window to exit)")
        # We need the root folder here to construct the full paths for display
        show_results(image_path, results, image_root_folder, top_n)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Find similar images in your dataset.")
    parser.add_argument("input_image", type=str, help="The full path to the new image you want to find matches for.")
    # The root folder is now needed again, but only for displaying the final results.
    parser.add_argument("image_folder", type=str, help="The root folder where your dataset images are stored for display.")
    parser.add_argument("--top_n", type=int, default=5, help="Number of similar images to display.")
    parser.add_argument("--db_name", type=str, default="images.db", help="The name of your SQLite database file.")
    # Add the flag to control the display
    parser.add_argument("--no-display", action="store_true", help="Run the script without showing the visual plot of results.")
    
    args = parser.parse_args()
    
    main(args.input_image, args.image_folder, args.top_n, args.db_name, args.no_display)