from einops import rearrange
import torch
from torch import nn, Tensor

class DecoderBlock(nn.Module):
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

class Transformer(nn.Module):
    def __init__(self, 
                 num_layers: int, 
                 dim_model: int, 
                 num_heads: int, 
                 num_tokens: int, 
                 seq_len: int, 
                 dropout: float, 
                 max_context_len: int = 1024):
        super().__init__()

        self.max_context_len = max_context_len
        self.token_embeddings = nn.Embedding(num_tokens, dim_model)
        self.position_embeddings = nn.Embedding(seq_len, dim_model)
        self.model = nn.Sequential(
            *[DecoderBlock(dim_model, num_heads, dropout) for _ in range(num_layers)],
            nn.LayerNorm(dim_model),
            nn.Linear(dim_model, num_tokens)
        )

    def forward(self, inputs: Tensor):
        batch_size, context_len = inputs.shape
        token_embedding = self.token_embeddings(inputs)
        position_embedding = self.position_embeddings(torch.arange(context_len, device=inputs.device).unsqueeze(0).repeat(batch_size, 1))
        embedding = token_embedding + position_embedding
        embedding = rearrange(embedding, 'b s d -> s b d')
        output = self.model(embedding)
        return output[-1, :, :]

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

class LSTM(nn.Module):
    def __init__(self,
                 num_layers: int,
                 dim_model: int,
                 num_tokens: int,
                 hidden_dim: int,
                 dropout: float):
        super(LSTM, self).__init__()

        self.token_embeddings = nn.Embedding(num_tokens, dim_model)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(dim_model, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, num_tokens)

    def forward(self, inputs: torch.Tensor):
        embedded = self.token_embeddings(inputs)
        lstm_out, _ = self.lstm(embedded)
        return self.fc(lstm_out[:,-1,:])
