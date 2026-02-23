import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    """
    Implementation of Multi-Head Attention from scratch.
    This is the core mechanism that allows the model to 'attend' to different parts of 
    a sequence simultaneously.
    """
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size  # The total dimension of the input word vector (e.g., 256)
        self.heads = heads            # Number of parallel attention 'heads' (e.g., 8)
        self.head_dim = embed_size // heads # Dimension of each individual head (e.g., 256/8 = 32)

        # The embedding size MUST be perfectly divisible by the number of heads
        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        # These linear layers transform the input into Query, Key, and Value vectors.
        # Note: We only define the linear transformation for a SINGLE head dimension, 
        # but we apply it to all heads at once later using reshaping.
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        
        # This final linear layer combines the outputs from all heads back into one vector.
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, queries, mask):
        # N: Number of examples in the batch (Batch Size)
        N = queries.shape[0]

        # Get lengths of the value, key, and query sequences
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        # 1. SPLIT: Reshape the input embeddings into (Batch, Seq_Len, Heads, Head_Dim)
        # This effectively creates multiple 'heads' that each look at a smaller part of the data.
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        # 2. TRANSFORM: Apply the linear layers defined in __init__
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # 3. ATTENTION SCORE (Scaled Dot-Product):
        # We calculate the 'energy' (how much each word in the query attends to each word in the key).
        # energy = (queries * keys_transpose) / sqrt(d_k)
        # Using 'einsum' (Einstein Summation) is a highly efficient way to do complex tensor multiplication.
        # n: batch, q: query_len, h: heads, d: head_dim, k: key_len
        # Resulting energy shape: (N, heads, query_len, key_len)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        # 4. MASKING (Optional):
        # If a mask is provided (like in the decoder), we fill '0' positions with a tiny number (-1e20).
        # When we apply softmax later, these positions will effectively become zero.
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # 5. SOFTMAX:
        # Normalize the scores so they sum to 1. This creates the 'Attention Weights'.
        # We divide by (embed_size ** 1/2) to scale the values and prevent gradients from exploding.
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        # 6. COMBINE WITH VALUES:
        # Multiply the attention weights by the 'Value' vectors to get the weighted output.
        # out shape: (N, query_len, heads, head_dim)
        out = torch.einsum("nhqk,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        # 7. FINAL PROJECTION:
        # Pass the concatenated head outputs through the final linear layer.
        return self.fc_out(out)
