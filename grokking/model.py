from einops import rearrange
import numpy as np
import torch
from torch import nn, Tensor

class DecoderBlock(torch.nn.Module):
    def __init__(self, dim_model: int, n_heads: int, dropout: float):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(dim_model, n_heads)
        self.self_attn_norm = nn.LayerNorm(dim_model)
        self.ffn = nn.Sequential(
            nn.Linear(dim_model, dim_model * 4),
            nn.GELU(),
            nn.Linear(dim_model * 4, dim_model)
        )
        self.ffn_drop = nn.Dropout(p=dropout)
        self.ffn_norm = nn.LayerNorm(dim_model)

    def forward(self, x: Tensor):
        attn_mask = torch.full(
            (len(x), len(x)), -float("Inf"), device=x.device, dtype=x.dtype
        )
        attn_mask = torch.triu(attn_mask, diagonal=1)
    
        a1, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        a1 = self.self_attn_norm (x + a1)
        a2 = self.ffn(a1)
        a2 = self.ffn_drop(a2)
        a2 = self.ffn_norm(a1 + a2)

        return a2

class Transformer(torch.nn.Module):
    def __init__(self, 
                 num_layers: int, 
                 dim_model: int, 
                 num_heads: int, 
                 num_tokens: int, 
                 dropout: float,
                 max_context_len: int = 5):
        super().__init__()

        self.max_context_len = max_context_len
        self.token_embeddings = nn.Embedding(num_tokens, dim_model)
        self.register_buffer(
            "position_encoding", self._position_encoding(max_context_len, dim_model)
        )
        self.model = nn.Sequential(
            *[DecoderBlock(dim_model, num_heads, dropout) for _ in range(num_layers)],
            nn.LayerNorm(dim_model),
            nn.Linear(dim_model, num_tokens)
        )

    @classmethod
    def _position_encoding(cls, context_len: int, d_model: int) -> Tensor:
        rows = [Tensor([np.sin(pos / (10000 ** (i / d_model))) if i % 2 == 0
                    else np.cos(pos / (10000 ** ((i - 1) / d_model))) for i in range(d_model)])
                for pos in range(context_len)
                ]
        stack = torch.stack(rows, dim=1)

        return stack.T
    
    def embed(self, indices: Tensor) -> Tensor:
        batch_size = indices.shape[0]
        context_len = indices.shape[-1]
        pe = self.position_encoding[:context_len, :]
        pe = pe.unsqueeze(0).repeat(batch_size, 1, 1)

        embedded = self.token_embeddings(indices)

        return pe + embedded

    def forward(self, inputs: Tensor):
        embedding = self.embed(inputs)
        embedding = rearrange(embedding, 'b s d -> s b d')

        return self.model(embedding)

class MLP(nn.Module):
    def __init__(self, num_layers: int, dim_model: int, num_heads: int, num_tokens: int, seq_len: int, dropout: float = 0.0):
        super().__init__()
        self.token_embeddings = nn.Embedding(num_tokens, dim_model)
        self.position_embeddings = nn.Embedding(seq_len, dim_model)
        input_dim = seq_len * dim_model
        hidden_dim = input_dim
        layers = []
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            *layers,
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_tokens),
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, inputs: Tensor):
        batch_size, context_len = inputs.shape
        token_embeddings = self.token_embeddings(inputs)
        position_embeddings = self.position_embeddings(torch.arange(context_len, device=inputs.device).unsqueeze(0).repeat(batch_size, 1))
        embedding = token_embeddings + position_embeddings
        embedding = self.dropout(embedding)
        embedding = rearrange(embedding, 'b s d -> b (s d)')
        output = self.model(embedding)
        return output