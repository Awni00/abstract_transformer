import math
import torch
from torch import nn, Tensor

class SinusoidalPositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2048):
        """
        module which adds a (non-trainable) sinusoidal positional encoding to the input tensor

        Parameters
        ----------
        d_model : int
            model dimension.
        dropout : float, optional
            dropout rate, by default 0.1
        max_len : int, optional
            maximum length to consider, by default 2048
        """

        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe[:, 0, :]
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """

        x = x + self.pe[:x.size(1)]
        return self.dropout(x)

class LearnedPositionalEmbeddings(nn.Module):

    def __init__(self, d_model: int, dropout : float = 0.1, max_len: int = 2048):
        """module which adds a learnable positionall embedding to the input tensor.

        Parameters
        ----------
        d_model : int
            model dimension.
        dropout : float, optional
            dropout rate, by default 0.1
        max_len : int, optional
            maximum length to consider, by default 2048
        """

        super().__init__()
        self.position_embeddings = nn.Embedding(max_len, d_model)
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """

        positions = torch.arange(0, x.size(1), dtype=torch.long, device=x.device)
        positional_embedding = self.position_embeddings(positions)
        x = x + positional_embedding
        return self.dropout(x)

# TODO: implement attn with relative positional embedding or RoPE
