import os
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm

# --- Configuration ---
# Using verified stable mirrors for Flickr30k
DATASET_URLS = {
    "images": "https://github.com/jbrownlee/Datasets/releases/download/Flickr30k/flickr30k_images.zip",
    "text": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/Flickr30k/results.csv", 
}

DATA_DIR = Path(__file__).parent.parent.parent / "data"
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

def extract_zip(zip_path: Path, extract_to: Path):
    """Extracts zip and deletes it."""
    print(f"Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    zip_path.unlink()

def main():
    print("--- Starting Flickr30k Preparation ---")
    DATA_DIR.mkdir(exist_ok=True)

    # 1. Download Text (results.csv)
    if not TEXT_FILE.exists():
        print("Downloading Flickr30k captions file...")
        download_file(DATASET_URLS["text"], TEXT_FILE)
        print("Text file ready.")
    else:
        print("Text file already exists.")

    # 2. Download Images (~4.1 GB)
    if not IMAGES_DIR.exists():
        img_zip = DATA_DIR / "flickr30k_images.zip"
        download_file(DATASET_URLS["images"], img_zip)
        extract_zip(img_zip, DATA_DIR)
        print("Images ready.")
    else:
        print("Images directory already exists.")

    print("\n--- Flickr30k Preparation Complete ---")

if __name__ == "__main__":
    main()
