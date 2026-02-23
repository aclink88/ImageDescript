import torch
import torch.nn as nn
from .transformer_block import TransformerBlock
from .decoder_layer import DecoderLayer

class CustomTransformer(nn.Module):
    """
    A full Encoder-Decoder Transformer implemented from scratch.
    This class orchestrates everything: Embeddings, Stacking Layers, and Masking.
    """
    def __init__(
        self,
        src_vocab_size,   # Number of unique tokens in source (e.g., image pixels or English words)
        trg_vocab_size,   # Number of unique tokens in target (e.g., French words or captions)
        src_pad_idx,      # The ID used for padding in source
        trg_pad_idx,      # The ID used for padding in target
        embed_size=256,   # Dimension of the word vectors
        num_layers=6,     # Number of Encoder and Decoder blocks to stack
        forward_expansion=4, 
        heads=8,
        dropout=0,
        device="cuda",
        max_length=100,   # Maximum sequence length allowed
    ):
        super(CustomTransformer, self).__init__()

        self.embed_size = embed_size
        self.device = device
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

        # --- WORD EMBEDDINGS ---
        # Converts integer token IDs into dense numerical vectors.
        self.src_word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embed_size)

        # --- POSITIONAL EMBEDDINGS ---
        # Since Transformers have no order, we learn a vector for every possible position (0 to max_length).
        # We add this to the word vector so the model knows where a word is in the sentence.
        self.src_position_embedding = nn.Embedding(max_length, embed_size)
        self.trg_position_embedding = nn.Embedding(max_length, embed_size)

        # --- ENCODER STACK ---
        # Stacks 'num_layers' of TransformerBlocks.
        self.encoder_layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, dropout, forward_expansion)
                for _ in range(num_layers)
            ]
        )

        # --- DECODER STACK ---
        # Stacks 'num_layers' of DecoderLayers.
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(embed_size, heads, forward_expansion, dropout)
                for _ in range(num_layers)
            ]
        )

        # Final output layer: Maps the embed_size back to the vocabulary size to predict the next word.
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def make_src_mask(self, src):
        """
        Creates a mask to ignore padding tokens in the source.
        Shape: (Batch, 1, 1, src_len)
        """
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        """
        Creates a complex mask for the decoder.
        It must ignore padding AND prevent the model from seeing future tokens.
        """
        N, trg_len = trg.shape
        # 1. Padding mask
        trg_padding_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        
        # 2. Look-ahead mask (Lower Triangular Matrix)
        # This creates a matrix of 1s and 0s where a word at position 'i' 
        # can only see words at position 'j' if j <= i.
        trg_no_lookahead_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        ).to(self.device)

        # Combine both: A word is visible only if it's NOT padding AND NOT in the future.
        trg_mask = trg_padding_mask & trg_no_lookahead_mask
        return trg_mask

    def forward(self, src, trg):
        # --- 1. ENCODER PASS ---
        src_seq_length = src.shape[1]
        # Generate position indices [0, 1, 2, ...]
        src_positions = torch.arange(0, src_seq_length).expand(src.shape[0], src_seq_length).to(self.device)
        # Add word + position embeddings
        out = self.dropout(self.src_word_embedding(src) + self.src_position_embedding(src_positions))
        
        src_mask = self.make_src_mask(src)
        # Pass through all stacked encoder layers
        for layer in self.encoder_layers:
            out = layer(out, out, out, src_mask)

        # --- 2. DECODER PASS ---
        trg_seq_length = trg.shape[1]
        trg_positions = torch.arange(0, trg_seq_length).expand(trg.shape[0], trg_seq_length).to(self.device)
        # Add word + position embeddings
        x = self.dropout(self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions))
        
        trg_mask = self.make_trg_mask(trg)
        # Pass through all stacked decoder layers
        for layer in self.decoder_layers:
            # Note: out (from encoder) is passed as Value and Key
            x = layer(x, out, out, src_mask, trg_mask)

        # --- 3. FINAL OUTPUT ---
        # Map back to vocab space to get probabilities for each word
        predictions = self.fc_out(x)
        return predictions
