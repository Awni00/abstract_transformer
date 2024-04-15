import math
import torch
from torch import nn, Tensor

class SinusoidalPositionalEncoding(nn.Module):

    def __init__(self, d_model: int, scale: bool = True, dropout: float = 0.1, max_len: int = 2048):
        """
        module which adds a (non-trainable) sinusoidal positional encoding to the input tensor

        Parameters
        ----------
        d_model : int
            model dimension.
        scale : bool, optional
            whether to scale added positional encodings to account for scaling in dot product attention,
            by default True
        dropout : float, optional
            dropout rate, by default 0.1
        max_len : int, optional
            maximum length to consider, by default 2048
        """

        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = math.sqrt(d_model) if scale else 1

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

        x = self.scale * x + self.pe[:x.size(1)]
        return self.dropout(x)

class LearnedPositionalEmbeddings(nn.Module):

    def __init__(self, d_model: int, scale: bool = True, dropout: float = 0.1, max_len: int = 2048):
        """module which adds a learnable positionall embedding to the input tensor.

        Parameters
        ----------
        d_model : int
            model dimension.
        scale : bool, optional
            whether to scale added positional encodings to account for scaling in dot product attention,
            by default True
        dropout : float, optional
            dropout rate, by default 0.1
        max_len : int, optional
            maximum length to consider, by default 2048
        """

        super().__init__()
        self.scale = math.sqrt(d_model) if scale else 1

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
        x = self.scale * x + positional_embedding
        return self.dropout(x)

class RelativePositionalEncoding(nn.Module):

    def __init__(self, dim: int, max_rel_pos: int):
        """
        module which returns relative positional embeddings for a given pair of sequences.

        I.e., returns tensor whose [i,j]-th entry is the embedding of the relative position "j-i"

        Parameters
        ----------
        dim : int
            dimension of embeddings
        max_rel_pos : int
            maximum relative position in either direction (used for clipping)
        """

        super().__init__()
        self.num_units = dim
        self.max_relative_position = max_rel_pos
        self.rel_pos_embeddings = nn.Parameter(torch.Tensor(max_rel_pos * 2 + 1, dim))
        nn.init.xavier_uniform_(self.rel_pos_embeddings)

    def forward(self, length_q, length_k=None):
        """
        Parameters
        ----------
        length_q : int
            length of query sequence
        length_k : _type_, optional
            length of key sequence, by default None

        Returns
        -------
        Tensor
            tensor of shape [length_q, length_k, dim] where [i,j] is the embedding of the relative position "j-i"
        """

        if length_k is None:
            length_k = length_q

        range_q = torch.arange(length_q) # TODO: need to set dtype or device?
        range_k = torch.arange(length_k)

        distance_mat = range_k[None, :] - range_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)

        final_mat = distance_mat_clipped + self.max_relative_position

        # final_mat = torch.LongTensor(final_mat).cuda()
        embeddings = self.rel_pos_embeddings[final_mat] #.cuda()

        return embeddings

