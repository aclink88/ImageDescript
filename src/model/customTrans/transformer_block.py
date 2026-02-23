import torch
import torch.nn as nn
from .attention import MultiHeadAttention

class TransformerBlock(nn.Module):
    """
    A single Transformer Encoder Block.
    This is the fundamental building block. It contains an Attention layer followed 
    by a Feed-Forward network, with 'Skip Connections' and 'Layer Normalization' 
    to ensure stable training.
    """
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        # The Multi-Head Attention mechanism we built
        self.attention = MultiHeadAttention(embed_size, heads)
        
        # Layer Normalization: Keeps activations in a healthy range
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        # Feed-Forward Network: A simple two-layer network applied to every position independently.
        # 'forward_expansion' (usually 4) makes the hidden layer 4 times larger than the input.
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        # Dropout: Randomly turns off neurons during training to prevent overfitting.
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        # 1. ATTENTION STEP
        attention = self.attention(value, key, query, mask)
        
        # 2. RESIDUAL CONNECTION + NORM 1
        # We add the original 'query' back to the attention output (Skip Connection).
        # This helps gradients flow through deep networks.
        x = self.dropout(self.norm1(attention + query))
        
        # 3. FEED-FORWARD STEP
        forward = self.feed_forward(x)
        
        # 4. RESIDUAL CONNECTION + NORM 2
        # Add the input of the FF network back to its output.
        out = self.dropout(self.norm2(forward + x))
        
        return out
