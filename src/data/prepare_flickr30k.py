import os
import requests
from pathlib import Path
from tqdm import tqdm
import subprocess

# --- Configuration ---
# Multi-part download URLs from the awsaf49 repository
DATASET_PARTS = [
    "https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr30k_part00",
    "https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr30k_part01",
    "https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr30k_part02",
]

DATA_DIR = Path(__file__).parent.parent.parent / "data"
# These should match the names inside the zip and our loader expectations
IMAGES_DIR = DATA_DIR / "flickr30k_images"
TEXT_FILE = DATA_DIR / "results.csv"

def download_file(url: str, destination: Path):
    """Downloads a file with a progress bar."""
    print(f"Downloading {url} to {destination}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            with open(destination, 'wb') as f, tqdm(
                desc=destination.name,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    bar.update(size)
    except Exception as e:
        print(f"Download error: {e}")
        if destination.exists(): destination.unlink()
        raise

def main():
    print("--- Starting Flickr30k Preparation (Multi-part) ---")
    DATA_DIR.mkdir(exist_ok=True)

    # 1. Check if already extracted
    if IMAGES_DIR.exists() and TEXT_FILE.exists():
        print("Dataset already exists and is extracted. Skipping.")
        return

    # 2. Download all parts
    part_files = []
    for i, url in enumerate(DATASET_PARTS):
        part_path = DATA_DIR / f"flickr30k_part{i:02d}"
        part_files.append(part_path)
        if not part_path.exists():
            download_file(url, part_path)
        else:
            print(f"Part {i} already exists.")

    # 3. Concatenate parts
    combined_zip = DATA_DIR / "flickr30k_combined.zip"
    if not combined_zip.exists():
        print("Combining parts into one zip file...")
        # Using binary write to combine files
        with open(combined_zip, 'wb') as outfile:
            for part_file in part_files:
                with open(part_file, 'rb') as infile:
                    outfile.write(infile.read())
        print("Combined zip created.")
    
    # 4. Extract
    print("Extracting dataset (~4GB). This will take a few minutes...")
    # Using subprocess for unzip -q which is faster for massive zips on Linux/Colab
    try:
        # -q for quiet, -o for overwrite, -d for destination
        subprocess.run(['unzip', '-q', '-o', str(combined_zip), '-d', str(DATA_DIR)], check=True)
        print("Extraction complete.")
    except Exception as e:
        print(f"Extraction failed: {e}. Please ensure 'unzip' is installed.")
        return

    # 5. Cleanup temporary zip files
    print("Cleaning up temporary files...")
    combined_zip.unlink()
    for part in part_files:
        part.unlink()

    print("\n--- Flickr30k Preparation Complete ---")
    print(f"Images: {IMAGES_DIR}")
    print(f"Captions: {TEXT_FILE}")

if __name__ == "__main__":
    main()
