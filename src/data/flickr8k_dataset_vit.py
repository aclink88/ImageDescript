import os
import torch
import random
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from transformers import GPT2Tokenizer, ViTImageProcessor
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset

class ImageCaptionDataset(Dataset):
    """
    Local Dataset class for Image Captioning (Flickr8k).
    """
    def __init__(self, root_dir, captions_file, tokenizer, image_processor, max_length=50, use_augmentation=False, dataset_type='flickr8k'):
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        self.use_augmentation = use_augmentation
        
        self.augmentation = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ])
        
        img_names = []
        caption_texts = []
        
        with open(captions_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            start_idx = 1 if 'image' in lines[0] else 0
            for line in lines[start_idx:]:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    img_id, caption = parts
                    clean_id = img_id.split('#')[0]
                    img_names.append(clean_id)
                    caption_texts.append(caption)
        
        self.df = pd.DataFrame({'image': img_names, 'caption': caption_texts})

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        try:
            img_id = self.df.iloc[index]['image']
            img_path = os.path.join(self.root_dir, img_id)
            image = Image.open(img_path).convert("RGB")
            if self.use_augmentation:
                image = self.augmentation(image)
            pixel_values = self.image_processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
            caption = self.df.iloc[index]['caption']
            tokenized = self.tokenizer.encode(f"<|endoftext|> {caption} <|endoftext|>", max_length=self.max_length, truncation=True)
            return pixel_values, torch.tensor(tokenized)
        except Exception:
            return self.__getitem__(random.randint(0, len(self.df) - 1))

class HFImageCaptionDataset(Dataset):
    """
    Modern Dataset class that loads data directly from Hugging Face Hub.
    """
    def __init__(self, hf_dataset, tokenizer, image_processor, max_length=50, use_augmentation=True):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        self.use_augmentation = use_augmentation
        
        self.augmentation = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        # HF datasets return PIL images directly
        image = item['image'].convert("RGB")
        
        if self.use_augmentation:
            image = self.augmentation(image)
            
        pixel_values = self.image_processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)

        # In nlphuji/flickr30k, 'caption' is a list of 5 strings. 
        # We pick one at random for each training step to increase variety!
        captions = item['caption']
        caption = random.choice(captions) if isinstance(captions, list) else captions
        
        tokenized = self.tokenizer.encode(
            f"<|endoftext|> {caption} <|endoftext|>", 
            max_length=self.max_length, 
            truncation=True
        )

        return pixel_values, torch.tensor(tokenized)

class CollateModern:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
    def __call__(self, batch):
        imgs = torch.stack([item[0] for item in batch])
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)
        return imgs, targets

def get_loader_hf(dataset_name, batch_size=64, num_workers=4, use_augmentation=True):
    """
    Creates a DataLoader from a Hugging Face dataset.
    """
    # Use a modern, script-free version of Flickr30k
    target_dataset = "HuggingFaceM4/Flickr30k" if dataset_name == "nlphuji/flickr30k" else dataset_name
    
    print(f"Loading '{target_dataset}' from Hugging Face...")
    # 'HuggingFaceM4/Flickr30k' uses 'train' split for the main data
    raw_dataset = load_dataset(target_dataset, split="train")
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    
    dataset = HFImageCaptionDataset(
        raw_dataset, 
        tokenizer, 
        image_processor, 
        use_augmentation=use_augmentation
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
        collate_fn=CollateModern(pad_idx=tokenizer.pad_token_id)
    ), dataset

def get_loader_modern(root_folder, annotation_file, dataset_type='flickr8k', batch_size=64, num_workers=4, use_augmentation=True):
    """Original local loader for Flickr8k."""
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    dataset = ImageCaptionDataset(root_folder, annotation_file, tokenizer, image_processor, use_augmentation=use_augmentation, dataset_type=dataset_type)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True, collate_fn=CollateModern(pad_idx=tokenizer.pad_token_id)), dataset
