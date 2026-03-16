import os
import zipfile
from pathlib import Path
import subprocess

def main():
    print("--- Starting Kaggle Flickr30k Preparation ---")
    
    # 1. Setup Kaggle Auth
    # This script assumes you have uploaded 'kaggle.json' to the /content directory in Colab
    kaggle_json = Path("/content/kaggle.json")
    if not kaggle_json.exists():
        print("Error: /content/kaggle.json not found.")
        print("Please download your API token from Kaggle Settings and upload it to Colab.")
        return

    # Move to the required hidden folder for the Kaggle CLI
    home_kaggle = Path.home() / ".kaggle"
    home_kaggle.mkdir(exist_ok=True)
    subprocess.run(["cp", str(kaggle_json), str(home_kaggle / "kaggle.json")])
    subprocess.run(["chmod", "600", str(home_kaggle / "kaggle.json")])

    # 2. Define Data Directory
    data_dir = Path(__file__).parent.parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    # 3. Download from Kaggle
    # adityajn105/flickr30k is a very stable version of the dataset
    print("Downloading flickr30k from Kaggle (~4.1 GB)...")
    try:
        subprocess.run([
            "kaggle", "datasets", "download", "-d", "adityajn105/flickr30k", 
            "-p", str(data_dir), "--unzip"
        ], check=True)
        print("Download and unzip successful.")
    except Exception as e:
        print(f"Kaggle command failed: {e}")
        return

    # 4. Standardize Folder Structure
    # Kaggle zips often name their folders 'Images' or 'images'
    # We want 'flickr30k_images' to be consistent with our training script
    old_img_dir = data_dir / "Images"
    new_img_dir = data_dir / "flickr30k_images"
    
    if old_img_dir.exists() and not new_img_dir.exists():
        print(f"Renaming {old_img_dir.name} to {new_img_dir.name}...")
        old_img_dir.rename(new_img_dir)

    print("\n--- Flickr30k Preparation Complete ---")
    print(f"Dataset location: {data_dir.resolve()}")

if __name__ == "__main__":
    main()
