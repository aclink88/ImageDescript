import torch
import torch.nn as nn
import torch.optim as optim
from transformers import ViTImageProcessor, GPT2Tokenizer
from pathlib import Path
from tqdm import tqdm
import os

# Import our modern model and generic dataset logic
from src.model.modern_captioner import ModernCaptioner
from src.data.flickr8k_dataset_vit import get_loader_modern, get_loader_hf

def train_modern():
    """
    Optimized training loop for Modern Multi-modal Captioner.
    Supports Flickr8k (local) and Flickr30k (Hugging Face).
    """
    # --- 1. SETTINGS & HYPERPARAMETERS ---
    # These strings are replaced dynamically by the Colab dashboard
    dataset_type = 'flickr8k' 
    num_epochs = 10
    batch_size = 64
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device} | Dataset: {dataset_type}")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)

    learning_rate = 5e-5 
    
    # --- 2. DATA LOADING ---
    if dataset_type == 'flickr8k':
        base_dir = Path(os.getcwd())
        image_dir = base_dir / "data" / "Flickr8k_Dataset"
        captions_file = base_dir / "data" / "Flickr8k.token.txt"
        
        train_loader, dataset = get_loader_modern(
            root_folder=image_dir,
            annotation_file=captions_file,
            dataset_type='flickr8k',
            batch_size=batch_size,
            num_workers=4,
            use_augmentation=True
        )
    else:
        # Use Hugging Face loader for Flickr30k
        train_loader, dataset = get_loader_hf(
            dataset_name="nlphuji/flickr30k",
            batch_size=batch_size,
            num_workers=4,
            use_augmentation=True
        )

    # --- 3. MODEL INITIALIZATION ---
    print(f"Initializing ModernCaptioner with Dual-LoRA (Rank 64)...")
    model = ModernCaptioner(vocab_size=vocab_size, rank=64).to(device)

    # --- 4. OPTIMIZER & SCALER ---
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    # --- 5. TRAINING LOOP ---
    model.train()
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        loop = tqdm(train_loader, total=len(train_loader))
        
        for imgs, captions in loop:
            imgs = imgs.to(device)
            captions = captions.to(device)

            # Use Automatic Mixed Precision (bfloat16) for H100 speed
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(imgs, captions)
                text_logits = logits[:, 196:-1, :] 
                targets = captions
                loss = criterion(text_logits.reshape(-1, vocab_size), targets.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item())

    # --- 6. SAVE ---
    print("\nTraining complete. Saving modern model...")
    torch.save(model.state_dict(), "modern_captioner.pth")
    print("Model saved as modern_captioner.pth")

if __name__ == "__main__":
    train_modern()
