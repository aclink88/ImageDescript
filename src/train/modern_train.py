import torch
import torch.nn as nn
import torch.optim as optim
from transformers import ViTImageProcessor, GPT2Tokenizer
from pathlib import Path
from tqdm import tqdm
import os
import random
import wandb

# Import our modern model and generic dataset logic
from src.model.modern_captioner import ModernCaptioner
from src.data.flickr8k_dataset_vit import get_loader_modern

def train_modern():
    """
    Optimized training loop for Modern Multi-modal Captioner with wandb visualization.
    """
    # --- 1. SETTINGS & HYPERPARAMETERS ---
    dataset_type = 'flickr8k' 
    num_epochs = 10
    batch_size = 64
    max_length = 80  # Prevents caption cut-offs
    learning_rate = 5e-5 
    rank = 64
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device} | Dataset: {dataset_type}")

    # --- 2. INITIALIZE WANDB ---
    wandb.init(
        project="modern-image-captioner",
        config={
            "learning_rate": learning_rate,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "rank": rank,
            "max_length": max_length,
            "dataset": dataset_type,
            "architecture": "ViT-GPT2-LoRA"
        }
    )

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)

    # --- 3. DATA LOADING ---
    base_dir = Path(os.getcwd())
    
    if dataset_type == 'flickr8k':
        image_dir = base_dir / "data" / "Flickr8k_Dataset"
        captions_file = base_dir / "data" / "Flickr8k.token.txt"
    else:
        image_dir = base_dir / "data" / "flickr30k_images"
        captions_file = base_dir / "data" / "captions.txt" 

    print(f"Loading dataset from: {image_dir}")
    train_loader, dataset = get_loader_modern(
        root_folder=image_dir,
        annotation_file=captions_file,
        dataset_type=dataset_type,
        batch_size=batch_size,
        num_workers=4,
        use_augmentation=True,
        max_length=max_length
    )

    # --- 4. MODEL INITIALIZATION ---
    print(f"Initializing ModernCaptioner with Dual-LoRA (Rank {rank})...")
    model = ModernCaptioner(vocab_size=vocab_size, rank=rank).to(device)
    wandb.watch(model, log="all")

    # --- 5. OPTIMIZER & CRITERION ---
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    # --- 6. TRAINING LOOP ---
    for epoch in range(num_epochs):
        model.train()
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        loop = tqdm(train_loader, total=len(train_loader))
        
        epoch_loss = 0
        for batch_idx, (imgs, captions) in enumerate(loop):
            imgs = imgs.to(device)
            captions = captions.to(device)

            # Use Automatic Mixed Precision (bfloat16)
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(imgs, captions)
                # Skip visual tokens (196) and align for next-token prediction
                text_logits = logits[:, 196:-1, :] 
                loss = criterion(text_logits.reshape(-1, vocab_size), captions.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            wandb.log({"batch_loss": loss.item()})
            loop.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(train_loader)
        wandb.log({"epoch_loss": avg_loss, "epoch": epoch + 1})

        # --- 7. END OF EPOCH VISUALIZATION ---
        log_validation_samples(model, dataset, tokenizer, device, epoch)

    # --- 8. SAVE & FINISH ---
    print("\nTraining complete. Saving modern model...")
    save_path = "modern_captioner.pth"
    torch.save(model.state_dict(), save_path)
    wandb.save(save_path)
    wandb.finish()
    print(f"Model saved and synced to wandb.")

def log_validation_samples(model, dataset, tokenizer, device, epoch):
    """
    Pick random images, generate captions via beam search, and log to wandb.
    """
    model.eval()
    samples = []
    # Log 3 samples per epoch to track qualitative progress
    for i in range(3):
        idx = random.randint(0, len(dataset) - 1)
        img_tensor, _ = dataset[idx]
        
        with torch.no_grad():
            # Using the specialized Beam Search method from ModernCaptioner
            generated_caption = model.generate_caption_beam(
                img_tensor.to(device), 
                tokenizer, 
                max_length=40
            )
        
        # Log to wandb Media tab
        img_log = wandb.Image(
            img_tensor, 
            caption=f"Epoch {epoch+1} Sample {i+1}: {generated_caption}"
        )
        samples.append(img_log)
    
    wandb.log({"visual_progress": samples})
    print(f"Logged validation samples to wandb.")

if __name__ == "__main__":
    train_modern()
