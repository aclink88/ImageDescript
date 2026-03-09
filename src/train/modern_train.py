import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import ViTImageProcessor, GPT2Tokenizer
from pathlib import Path
from tqdm import tqdm
import os

# Import our modern model and existing dataset logic
from src.model.modern_captioner import ModernCaptioner
from src.data.flickr8k_dataset_vit import get_loader_vit

def train_modern():
    """
    Training loop for the Modern Multi-modal Captioner (ViT + GPT-2 + LoRA).
    """
    # --- 1. SETTINGS & HYPERPARAMETERS ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # GPT-2 Tokenizer for vocab size and padding
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)

    # Training Hyperparameters
    batch_size = 16 
    learning_rate = 5e-5 
    num_epochs = 5
    
    # --- 2. DATA LOADING ---
    print("Loading Flickr8k dataset (Modern)...")
    base_dir = Path(os.getcwd())
    image_dir = base_dir / "data" / "Flickr8k_Dataset"
    captions_file = base_dir / "data" / "Flickr8k.token.txt"

    train_loader, dataset = get_loader_vit(
        root_folder=image_dir,
        annotation_file=captions_file,
        batch_size=batch_size,
        num_workers=0
    )

    # --- 3. MODEL INITIALIZATION ---
    print("Initializing ModernCaptioner with Dual-LoRA (Rank 64)...")
    model = ModernCaptioner(
        vocab_size=vocab_size, 
        rank=64, 
        use_encoder_lora=True
    ).to(device)

    # --- 4. OPTIMIZER ---
    # We only optimize the parameters that require gradients (the LoRA adapters and the Bridge)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # --- 5. TRAINING LOOP ---
    model.train()
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        loop = tqdm(train_loader, total=len(train_loader))
        
        for imgs, captions in loop:
            imgs = imgs.to(device)
            # 'captions' from our old loader are tensors of word IDs.
            # We need to ensure they match GPT-2's format if we were doing this perfectly,
            # but for this learning phase, we'll assume our loader's IDs are compatible 
            # with the model's resized embedding layer.
            captions = captions.to(device)

            # Forward Pass
            # logits shape: (batch, 197 + seq_len, vocab_size)
            logits = model(imgs, captions)

            # --- THE ALIGNMENT MATH ---
            # Sequence: [Image(197 tokens)] + [Text(N tokens)]
            # Position 196 (last image patch) predicts Text[0]
            # Position 196 + N - 1 predicts Text[N-1]
            
            # Slice logits to get the N predictions for the text
            text_logits = logits[:, 196:-1, :] 
            
            # The targets are the full caption (length N)
            targets = captions

            # Calculate Loss
            loss = criterion(
                text_logits.reshape(-1, vocab_size), 
                targets.reshape(-1)
            )

            # Backward Pass
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
