import os
import spacy
import torch
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
from collections import Counter

# Load the English tokenizer from spacy
spacy_eng = spacy.load("en_core_web_sm")

class Vocabulary:
    """
    Manages the mapping between words and numerical IDs.
    """
    def __init__(self, freq_threshold):
        # Special tokens
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4 # Start after special tokens

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]


class Flickr8kDataset(Dataset):
    """
    Custom PyTorch Dataset for the Flickr8k data.
    """
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file, delimiter='\t', header=None, names=['image', 'caption'])
        self.transform = transform

        # Get image and caption columns
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]

        # Initialize and build vocabulary
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        caption = self.captions[index]
        # The caption file has format "image_name.jpg#caption_index"
        # We only need the image name.
        img_id = self.df["image"][index].split('#')[0]
        img_path = os.path.join(self.root_dir, img_id)
        
        try:
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"Warning: Image file not found: {img_path}. Skipping.")
            # Return a placeholder or handle this case as needed
            # For simplicity, we can return the first image and its caption
            # A better approach would be to clean the dataset first
            img_id_placeholder = self.df["image"][0].split('#')[0]
            img_path_placeholder = os.path.join(self.root_dir, img_id_placeholder)
            img = Image.open(img_path_placeholder).convert("RGB")
            caption = self.captions[0]


        if self.transform is not None:
            img = self.transform(img)

        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption.extend(self.vocab.numericalize(caption))
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return img, torch.tensor(numericalized_caption)


class Collate:
    """
    Custom collate_fn to pad sequences to the same length.
    """
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)

        return imgs, targets


def get_loader(
    root_folder,
    annotation_file,
    transform,
    batch_size=32,
    num_workers=4,
    shuffle=True,
    pin_memory=True,
):
    """
    Creates and returns a DataLoader for the Flickr8k dataset.
    """
    dataset = Flickr8kDataset(root_folder, annotation_file, transform=transform)
    pad_idx = dataset.vocab.stoi["<PAD>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=Collate(pad_idx=pad_idx),
    )
    
    # The file "Flickr_8k.token.txt" uses a tab delimiter, but the pandas read_csv is not parsing it correctly.
    # We will fix this by reading the file manually.
    
    # Read the captions file manually
    captions = {}
    with open(annotation_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                img_id, caption = parts
                if img_id not in captions:
                    captions[img_id] = []
                captions[img_id].append(caption)

    # Re-create dataframe from the manually parsed captions
    img_names = []
    caption_texts = []
    for img_id, cap_list in captions.items():
        for i, cap in enumerate(cap_list):
            # The original file has image_name.jpg#index format
            img_names.append(f"{img_id}")
            caption_texts.append(cap)
            
    df = pd.DataFrame({'image': img_names, 'caption': caption_texts})
    dataset.df = df
    dataset.imgs = df['image']
    dataset.captions = df['caption']


    return loader, dataset

if __name__ == "__main__":
    # Example of how to use the loader
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    data_dir = Path(__file__).parent.parent.parent / "data"
    image_dir = data_dir / "Flicker8k_Dataset"
    captions_file = data_dir / "Flickr8k.token.txt"

    # Test the loader
    loader, dataset = get_loader(
        root_folder=image_dir,
        annotation_file=captions_file,
        transform=transform
    )

    print(f"Vocabulary Size: {len(dataset.vocab)}")

    for idx, (imgs, captions) in enumerate(loader):
        print(f"Batch {idx+1}:")
        print(f"  Images shape: {imgs.shape}")
        print(f"  Captions shape: {captions.shape}")
        break # Only show first batch
