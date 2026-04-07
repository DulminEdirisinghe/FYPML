import os
import time
import random
import shutil
from datetime import datetime

# Import your matching dataset class
from src.bench_dataset import DronePairDataset

# ============== CONFIG ==============
# Folders containing your actual images (update these to your real paths)
SOURCE_FOLDER_A = 'data/data/SCD_Images/test'
SOURCE_FOLDER_B = 'data/data/YOLO_data/stationary/test'

# The folders the detector script is actively watching
STREAM_FOLDER_A = 'runs/folder_a'
STREAM_FOLDER_B = 'runs/folder_b'

MAX_IMAGES_TO_KEEP = 2  # Prevent folder from getting cluttered
STREAM_DELAY = 7.0      # Seconds between new images (simulates frame rate)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def cleanup_old_files(folder_path):
    """Keeps only the newest 'MAX_IMAGES_TO_KEEP' files in the folder."""
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
    
    if len(files) > MAX_IMAGES_TO_KEEP:
        # Sort files by creation time, oldest first
        files.sort(key=os.path.getctime)
        
        # Delete the oldest files until we reach our max limit
        files_to_delete = files[:-MAX_IMAGES_TO_KEEP]
        for f in files_to_delete:
            try:
                os.remove(f)
            except Exception as e:
                print(f"Could not delete {f}: {e}")

def main():
    print("Starting Stream Simulator...")
    
    ensure_dir(STREAM_FOLDER_A)
    ensure_dir(STREAM_FOLDER_B)

    print("Building Dataset to find matching pairs...")
    # Initialize your dataset to correctly pair Folder A and Folder B images
    dataset = DronePairDataset(SOURCE_FOLDER_A, SOURCE_FOLDER_B)

    if len(dataset) == 0:
        print("Error: No matched pairs found in the dataset. Please check your source paths.")
        return

    print(f"Found {len(dataset)} perfectly matched image pairs.")
    print(f"Simulating stream at 1 frame every {STREAM_DELAY} seconds...\n")

    try:
        while True:
            # 1. Pick a random, MATCHED pair from the dataset
            random_idx = random.randint(0, len(dataset) - 1)
            data = dataset[random_idx]

            src_path_a = data['a_img_path']
            src_path_b = data['b_img_path']

            # 2. Create a unified filename based on the current time
            timestamp = int(time.time() * 1000)
            common_filename = f"frame_{timestamp}.png"

            dest_path_a = os.path.join(STREAM_FOLDER_A, common_filename)
            dest_path_b = os.path.join(STREAM_FOLDER_B, common_filename)

            # 3. Copy files to the stream folders
            shutil.copy(src_path_a, dest_path_a)
            shutil.copy(src_path_b, dest_path_b)

            # Output what was just sent to the stream
            original_a_name = os.path.basename(src_path_a)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Streamed pair: {common_filename} (Original: {original_a_name})")

            # 4. Clean up old files to avoid clutter
            cleanup_old_files(STREAM_FOLDER_A)
            cleanup_old_files(STREAM_FOLDER_B)

            # 5. Wait before sending the next frame
            time.sleep(STREAM_DELAY)

    except KeyboardInterrupt:
        print("\nStream simulator stopped.")

if __name__ == "__main__":
    main()