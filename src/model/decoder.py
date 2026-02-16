import torch
import torch.nn as nn

class DecoderRNN(nn.Module):
    """
    A Transformer-based Decoder for generating captions.
    """
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, nhead=8):
        """
        Args:
            embed_size (int): The size of the embedding vector from the encoder.
            hidden_size (int): The dimension of the feedforward network model in the transformer.
            vocab_size (int): The size of the vocabulary.
            num_layers (int): The number of sub-decoder-layers in the decoder.
            nhead (int): The number of heads in the multiheadattention models.
        """
        super(DecoderRNN, self).__init__()
        
        # Word embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # Positional Encoding (can be learned or fixed, here we use a simple learnable one)
        # Note: A more traditional implementation might use sinusoidal positional encodings.
        self.positional_encoding = nn.Embedding(100, embed_size) # Max caption length of 100
        
        # --- Transformer Decoder ---
        # Create a single Transformer Decoder Layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size, 
            nhead=nhead, 
            dim_feedforward=hidden_size,
            batch_first=True # Our data will be (batch, seq, feature)
        )
        # Stack multiple decoder layers
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Final fully connected layer to map to vocabulary space
        self.fc = nn.Linear(embed_size, vocab_size)
        
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        """
        Forward pass for the decoder.

        Args:
            features (torch.Tensor): Feature vectors from the encoder. Shape: (batch_size, embed_size)
            captions (torch.Tensor): Ground truth captions. Shape: (batch_size, seq_length)
        """
        # 1. Get embeddings for the captions and add positional encoding
        seq_length = captions.shape[1]
        positions = torch.arange(0, seq_length).expand(captions.shape[0], seq_length).to(captions.device)
        captions_embedded = self.dropout(self.embedding(captions) + self.positional_encoding(positions))

        # 2. Reshape encoder features to be the 'memory' for the decoder
        # TransformerDecoder expects memory as (seq_len, batch, feature), so we add a sequence dimension of 1
        # UPDATE for batch_first=True: memory should be (batch, seq_len, feature)
        memory = features.unsqueeze(1)

        # 3. Create masks for the decoder
        # Target mask: to prevent the decoder from looking at future words in the sequence
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_length).to(captions.device)

        # 4. Pass through the transformer decoder
        # The 'tgt' is the caption sequence, and 'memory' is the image features
        outputs = self.transformer_decoder(
            tgt=captions_embedded,
            memory=memory,
            tgt_mask=tgt_mask
        )
        
        # 5. Pass the output through the final linear layer
        predictions = self.fc(outputs)
        
        return predictions

if __name__ == '__main__':
    # A simple test to verify the decoder's output shape
    embed_size = 256
    hidden_size = 512
    vocab_size = 10000
    num_layers = 2
    
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    # Create dummy inputs
    batch_size = 5
    seq_length = 20
    
    # Dummy features from encoder
    features = torch.randn(batch_size, embed_size)
    # Dummy captions (batch of integer sequences)
    captions = torch.randint(0, vocab_size, (batch_size, seq_length))

    # Get the predictions
    predictions = decoder(features, captions)
    
    print(f"Input features shape: {features.shape}")
    print(f"Input captions shape: {captions.shape}")
    print(f"Output predictions shape: {predictions.shape}")
    print(f"Expected output shape: {(batch_size, seq_length, vocab_size)}")
    
    assert predictions.shape == (batch_size, seq_length, vocab_size)
    print("\nDecoder test passed!")
