import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .transformer_block import TransformerBlock

class DecoderLayer(nn.Module):
    """
    A single Transformer Decoder Block.
    The Decoder is special because it attends to TWO things:
    1. The words it has already generated (Masked Self-Attention).
    2. The output of the Encoder (Cross-Attention).
    """
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(DecoderLayer, self).__init__()
        self.norm = nn.LayerNorm(embed_size)
        self.attention = MultiHeadAttention(embed_size, heads)
        
        # We reuse the TransformerBlock class for the Cross-Attention and FF steps.
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        """
        Args:
            x: Input from the previous decoder layer or embedding.
            value, key: The 'Memory' from the Encoder (image features or source text).
            src_mask: Mask for the source (ignores padding).
            trg_mask: Mask for the target (prevents looking at future words).
        """
        # STEP 1: MASKED SELF-ATTENTION
        # The decoder looks at itself (x, x, x). 
        # The trg_mask ensures it can't see 'future' words.
        attention = self.attention(x, x, x, trg_mask)
        
        # Residual connection + Normalization
        query = self.dropout(self.norm(attention + x))
        
        # STEP 2: CROSS-ATTENTION (THE BRIDGE)
        # This is where the magic happens. The 'Query' comes from the Decoder (the words so far),
        # but the 'Key' and 'Value' come from the Encoder (the image/source text).
        # This allows the decoder to ask: 'Based on the words I've said, what should I look at in the image?'
        out = self.transformer_block(value, key, query, src_mask)
        
        return out
