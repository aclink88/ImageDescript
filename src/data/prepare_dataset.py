import os
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm

# --- Configuration ---
# Using a known-good public mirror of the dataset
DATASET_URLS = {
    "images": "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip",
    "text": "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Text.zip", # Corrected URL
}
# The destination directory for the data
DATA_DIR = Path(__file__).parent.parent.parent / "data"
IMAGES_DIR = DATA_DIR / "Flicker8k_Dataset"
TEXT_FILES_DIR = DATA_DIR / "Flickr8k_text"


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
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        # Clean up partial file
        if destination.exists():
            destination.unlink()
        raise
    except Exception as e:
        print(f"An unexpected error occurred during download: {e}")
        if destination.exists():
            destination.unlink()
        raise


def extract_zip(zip_path: Path, extract_to: Path):
    """Extracts a zip file and then deletes it."""
    print(f"Extracting {zip_path.name}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        # Clean up the zip file after extraction
        zip_path.unlink()
        print(f"Successfully extracted and removed {zip_path.name}.")
    except zipfile.BadZipFile:
        print(f"Error: {zip_path.name} is not a valid zip file or is corrupted.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred during extraction: {e}")
        raise

def main():
    """
    Main function to download and prepare the Flickr8k dataset.
    """
    print("--- Starting Dataset Preparation for Flickr8k ---")
    DATA_DIR.mkdir(exist_ok=True)

    # --- Step 1: Download and Extract Text/Captions ---
    if TEXT_FILES_DIR.exists():
        print("Text data directory already exists. Skipping download.")
    else:
        text_zip_path = DATA_DIR / "Flickr8k_text.zip"
        download_file(DATASET_URLS["text"], text_zip_path)
        extract_zip(text_zip_path, DATA_DIR)

    # --- Step 2: Download and Extract Images ---
    # This is the large file (~1 GB)
    if IMAGES_DIR.exists():
        print("Images directory already exists. Skipping download.")
    else:
        images_zip_path = DATA_DIR / "Flickr8k_Dataset.zip"
        download_file(DATASET_URLS["images"], images_zip_path)
        extract_zip(images_zip_path, DATA_DIR)
        # The zip extracts to 'Flicker8k_Dataset', a typo in the original zip.
        # Let's rename it to be consistent.
        if (DATA_DIR / "Flicker8k_Dataset").exists():
            (DATA_DIR / "Flicker8k_Dataset").rename(IMAGES_DIR)


    print("\n--- Dataset Preparation Complete ---")
    print(f"Images should be in: {IMAGES_DIR}")
    print(f"Caption files should be in: {TEXT_FILES_DIR}")


if __name__ == "__main__":
    main()
