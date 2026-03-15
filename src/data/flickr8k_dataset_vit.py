import os
import torch
import random
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from transformers import GPT2Tokenizer, ViTImageProcessor
from torch.nn.utils.rnn import pad_sequence

class ImageCaptionDataset(Dataset):
    """
    Generic Dataset class for Image Captioning datasets (Flickr8k, Flickr30k).
    Uses Hugging Face tokenizers and image processors.
    """
    def __init__(self, root_dir, captions_file, tokenizer, image_processor, max_length=50, use_augmentation=False, dataset_type='flickr8k'):
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        self.use_augmentation = use_augmentation
        
        # Data Augmentation pipeline
        self.augmentation = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ])
        
        # Load and parse captions
        img_names = []
        caption_texts = []
        
        with open(captions_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # Skip header if it exists
            start_idx = 1 if 'image' in lines[0] else 0
            
            for line in lines[start_idx:]:
                if dataset_type == 'flickr8k':
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        img_id, caption = parts
                        # Clean artifacts
                        clean_id = img_id.split('#')[0]
                        if clean_id.endswith('.1') or clean_id.endswith('.2'):
                            clean_id = clean_id.rsplit('.', 1)[0]
                        img_names.append(clean_id)
                        caption_texts.append(caption)
                
                elif dataset_type == 'flickr30k':
                    # Flickr30k typically uses CSV format: image_name|comment_number|comment
                    parts = line.strip().split('|')
                    if len(parts) == 3:
                        img_id, _, caption = parts
                        img_names.append(img_id.strip())
                        caption_texts.append(caption.strip())
        
        self.df = pd.DataFrame({'image': img_names, 'caption': caption_texts})
        print(f"Loaded {len(self.df)} image-caption pairs for {dataset_type}.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        try:
            # 1. Process Image
            img_id = self.df.iloc[index]['image']
            img_path = os.path.join(self.root_dir, img_id)
            image = Image.open(img_path).convert("RGB")
            
            # Apply Augmentation if in training mode
            if self.use_augmentation:
                image = self.augmentation(image)
            
            pixel_values = self.image_processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)

            # 2. Process Caption
            caption = self.df.iloc[index]['caption']
            tokenized = self.tokenizer.encode(
                f"<|endoftext|> {caption} <|endoftext|>", 
                max_length=self.max_length, 
                truncation=True
            )

            return pixel_values, torch.tensor(tokenized)
            
        except Exception as e:
            # Graceful replacement on load failure
            new_index = random.randint(0, len(self.df) - 1)
            return self.__getitem__(new_index)

class CollateModern:
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

def get_loader_modern(
    root_folder,
    annotation_file,
    dataset_type='flickr8k',
    batch_size=64,
    num_workers=4,
    use_augmentation=True
):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    
    dataset = ImageCaptionDataset(
        root_dir=root_folder,
        captions_file=annotation_file,
        tokenizer=tokenizer,
        image_processor=image_processor,
        use_augmentation=use_augmentation,
        dataset_type=dataset_type
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
        collate_fn=CollateModern(pad_idx=tokenizer.pad_token_id)
    )

    return loader, dataset
