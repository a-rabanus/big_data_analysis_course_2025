import os
import pathlib

def simple_path_generator(root_folder, processed_paths_set, batch_size=128):
    """
    Finds all unprocessed image paths and yields them in batches of paths.
    """
    paths_to_process = []
    image_extensions = {".jpg", ".jpeg", ".png"}

    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            full_path = os.path.normpath(os.path.join(dirpath, filename))

            if pathlib.Path(full_path).suffix.lower() in image_extensions:
                if full_path not in processed_paths_set:
                    paths_to_process.append(full_path)

            if len(paths_to_process) == batch_size:
                yield paths_to_process
                paths_to_process = []

    if paths_to_process:
        yield paths_to_process