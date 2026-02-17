import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import os
from pathlib import Path
from tqdm import tqdm

# Import our custom modules
from src.data.flickr8k_dataset import get_loader
from src.model.captioning_model import CaptioningModel

def train():
    """
    Main training loop for the image captioning model.
    """
    # --- Hyperparameters ---
    embed_size = 256
    hidden_size = 512
    vocab_size = None # Will be set by the dataset
    num_layers = 2
    learning_rate = 3e-4
    num_epochs = 10 # Increase for better results
    
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((356, 356)),
        transforms.RandomCrop((299, 299)), # InceptionV3 input size
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # --- Data Loading ---
    print("Loading data...")
    # Define paths
    # Use absolute path from current working directory
    base_dir = Path(os.getcwd())
    data_dir = base_dir / "data"
    image_dir = data_dir / "Flickr8k_Dataset" 
    captions_file = data_dir / "Flickr8k.token.txt"

    # Get data loader
    # Note: num_workers > 0 can cause issues on Windows. Set to 0 if you have problems.
    train_loader, dataset = get_loader(
        root_folder=image_dir,
        annotation_file=captions_file,
        transform=transform,
        num_workers=0,
        batch_size=32
    )
    
    # Update vocab_size from the loaded dataset
    vocab_size = len(dataset.vocab)
    
    # --- Model Initialization ---
    print("Initializing model...")
    model = CaptioningModel(
        embed_size=embed_size,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_layers=num_layers
    ).to(device)

    # --- Loss and Optimizer ---
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # --- Training Loop ---
    print("Starting training...")
    model.train() # Set model to training mode

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        
        # Using tqdm for a progress bar
        loop = tqdm(train_loader, total=len(train_loader), leave=True)
        
        for idx, (imgs, captions) in enumerate(loop):
            imgs = imgs.to(device)
            captions = captions.to(device)

            # Forward pass
            outputs = model(imgs, captions[:-1]) # Don't feed <EOS> token to decoder

            # Reshape for loss calculation
            # outputs: (seq_len, batch_size, vocab_size)
            # captions: (seq_len, batch_size)
            # We want to compare each word in the output sequence to the target sequence
            # The last token of input captions is <EOS>, so we compare outputs with captions from the 1st token
            loss = criterion(
                outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1)
            )

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update progress bar
            loop.set_postfix(loss=loss.item())
            
    # --- Save Model ---
    print("\nTraining complete. Saving model...")
    model_path = base_dir / "captioning_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    train()
