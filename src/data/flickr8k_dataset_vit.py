import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from transformers import GPT2Tokenizer, ViTImageProcessor
from torch.nn.utils.rnn import pad_sequence

class Flickr8kDatasetViT(Dataset):
    """
    A Modern Dataset class for Flickr8k optimized for ViT and GPT-2.
    Uses Hugging Face tokenizers and image processors.
    """
    def __init__(self, root_dir, captions_file, tokenizer, image_processor, max_length=50):
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        
        # Read the captions file (manually to handle the tab delimiter correctly)
        img_names = []
        caption_texts = []
        with open(captions_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    img_id, caption = parts
                    # Format is 'image_name.jpg#0'
                    img_names.append(img_id.split('#')[0])
                    caption_texts.append(caption)
        
        self.df = pd.DataFrame({'image': img_names, 'caption': caption_texts})

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # 1. Process Image
        img_id = self.df.iloc[index]['image']
        img_path = os.path.join(self.root_dir, img_id)
        image = Image.open(img_path).convert("RGB")
        
        # ViT processor handles resizing and normalization
        pixel_values = self.image_processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)

        # 2. Process Caption
        caption = self.df.iloc[index]['caption']
        
        # Use GPT-2 Tokenizer
        # We add SOS and EOS tokens (GPT-2 usually uses <|endoftext|> for both)
        tokenized = self.tokenizer.encode(
            f"<|endoftext|> {caption} <|endoftext|>", 
            max_length=self.max_length, 
            truncation=True
        )

        return pixel_values, torch.tensor(tokenized)

class CollateViT:
    """
    Handles batching and padding for the ViT+GPT model.
    """
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = torch.stack([item[0] for item in batch])
        # Pad captions to the same length in the batch
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)

        return imgs, targets

def get_loader_vit(
    root_folder,
    annotation_file,
    batch_size=16,
    num_workers=0,
    shuffle=True
):
    """
    Creates a DataLoader for the ViT+GPT model.
    """
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    
    dataset = Flickr8kDatasetViT(
        root_dir=root_folder,
        captions_file=annotation_file,
        tokenizer=tokenizer,
        image_processor=image_processor
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=CollateViT(pad_idx=tokenizer.pad_token_id)
    )

    return loader, dataset

if __name__ == "__main__":
    # Test the modern loader
    from pathlib import Path
    data_dir = Path("data")
    image_dir = data_dir / "Flickr8k_Dataset"
    captions_file = data_dir / "Flickr8k.token.txt"
    
    if image_dir.exists():
        loader, dataset = get_loader_vit(image_dir, captions_file)
        print(f"Dataset Size: {len(dataset)}")
        imgs, caps = next(iter(loader))
        print(f"Images batch shape: {imgs.shape}")
        print(f"Captions batch shape: {caps.shape}")
    else:
        print("Data directory not found. Please run the preparation script first.")
