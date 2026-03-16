import os
import zipfile
from pathlib import Path
import subprocess

def main():
    print("--- Starting Kaggle Flickr30k Preparation ---")
    
    # 1. Define Data Directory
    data_dir = Path(__file__).parent.parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    # 2. Download from Kaggle
    # The 'kaggle' library automatically looks in ~/.kaggle/kaggle.json
    print("Downloading and unzipping flickr30k from Kaggle (~4.1 GB)...")
    try:
        # Check if already exists to save time
        if (data_dir / "results.csv").exists():
            print("Dataset files already detected. Skipping download.")
        else:
            subprocess.run([
                "kaggle", "datasets", "download", "-d", "adityajn105/flickr30k", 
                "-p", str(data_dir), "--unzip"
            ], check=True)
            print("Download and unzip successful.")
    except Exception as e:
        print(f"Kaggle command failed: {e}")
        print("Tip: Ensure your Kaggle API credentials are correct.")
        return

    # 3. Standardize Folder Structure
    # Kaggle zips often name their folders 'Images' or 'images'
    old_img_dir = data_dir / "Images"
    new_img_dir = data_dir / "flickr30k_images"
    
    if old_img_dir.exists() and not new_img_dir.exists():
        print(f"Renaming {old_img_dir.name} to {new_img_dir.name}...")
        old_img_dir.rename(new_img_dir)

    print("\n--- Flickr30k Preparation Complete ---")
    print(f"Dataset location: {data_dir.resolve()}")

if __name__ == "__main__":
    main()
