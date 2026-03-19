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
    Optimized training loop for Modern Multi-modal Captioner with wandb.Table monitoring.
    """
    # --- 1. SETTINGS & HYPERPARAMETERS ---
    dataset_type = 'flickr8k' 
    num_epochs = 10
    batch_size = 64
    max_length = 80 
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
            "dataset": dataset_type
        }
    )

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)

    # --- 3. DATA LOADING ---
    base_dir = Path(os.getcwd())
    image_dir = base_dir / "data" / "Flickr8k_Dataset" if dataset_type == 'flickr8k' else base_dir / "data" / "flickr30k_images"
    captions_file = base_dir / "data" / "Flickr8k.token.txt" if dataset_type == 'flickr8k' else base_dir / "data" / "captions.txt"

    train_loader, dataset = get_loader_modern(
        root_folder=image_dir,
        annotation_file=captions_file,
        dataset_type=dataset_type,
        batch_size=batch_size,
        num_workers=4,
        use_augmentation=True,
        max_length=max_length
    )

    # --- 4. PREPARE FIXED MONITORING SAMPLES ---
    fixed_indices = [10, 100, 500] 
    fixed_samples = []
    for idx in fixed_indices:
        img_tensor, cap_tensor = dataset[idx]
        gt_text = tokenizer.decode(cap_tensor, skip_special_tokens=True).strip()
        fixed_samples.append({
            "image": img_tensor.to(device),
            "gt": gt_text
        })

    # --- 5. MODEL INITIALIZATION ---
    model = ModernCaptioner(vocab_size=vocab_size, rank=rank).to(device)
    wandb.watch(model, log="all")

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    # --- 6. TRAINING LOOP ---
    global_step = 0
    for epoch in range(num_epochs):
        model.train()
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        loop = tqdm(train_loader, total=len(train_loader))
        
        for batch_idx, (imgs, captions) in enumerate(loop):
            imgs, captions = imgs.to(device), captions.to(device)

            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(imgs, captions)
                text_logits = logits[:, 196:-1, :] 
                loss = criterion(text_logits.reshape(-1, vocab_size), captions.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            
            # --- FIXED IMAGE MONITORING (TABLE) ---
            if batch_idx % 50 == 0:
                model.eval()
                # Create a Table to store our results for this step
                # Tables prevent the text cutoff seen in standard image captions
                results_table = wandb.Table(columns=["Step", "Image", "Ground Truth", "Prediction"])
                
                with torch.no_grad():
                    for sample in fixed_samples:
                        pred_text = model.generate_caption_beam(
                            sample["image"], 
                            tokenizer, 
                            max_length=max_length # Use the full max_length
                        )
                        results_table.add_data(
                            global_step,
                            wandb.Image(sample["image"].cpu()),
                            sample["gt"],
                            pred_text
                        )
                
                wandb.log({
                    "batch_loss": loss.item(),
                    "evaluation_table": results_table
                })
                model.train()
            else:
                wandb.log({"batch_loss": loss.item()})
                
            loop.set_postfix(loss=loss.item())

    # --- 7. SAVE & FINISH ---
    torch.save(model.state_dict(), "modern_captioner.pth")
    wandb.save("modern_captioner.pth")
    wandb.finish()
    print(f"Training complete. All results saved to wandb Table.")

if __name__ == "__main__":
    train_modern()
