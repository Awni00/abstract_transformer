import torch
import torch.nn as nn
from positional_encoding import RelativePositionalEncoding

class SymbolicAttention(nn.Module):
    def __init__(self,
            d_model: int,
            n_heads: int,
            n_symbols: int,
            dropout: float = 0.0,
            scale: float = None):
        """
        Symbolic Attention.

        Learns a library of "symbols" and corresponding template features.
        For a given input, retrieves a symbol from the symbol library via attention.

        Parameters
        ----------
        d_model : int
            model dimension. this is the dimension of the input and the dimension of the symbols and template features.
        n_heads : int
            number of heads in symbolic attention.
        n_symbols : int
            number of symbols in the symbol library.
        dropout : float, optional
            dropout probability, by default 0.0
        scale : float, optional
            scaling factor in scaled_dot_product_attention, by default None
        """

        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_symbols = n_symbols
        self.dropout = dropout
        self.scale = scale

        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.template_features = nn.Parameter(torch.empty(self.n_symbols, self.d_model))
        self.symbol_library = nn.Parameter(torch.empty(self.n_symbols, self.d_model))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.template_features)
        torch.nn.init.normal_(self.symbol_library)


    def forward(self, x):

        batch_size, seq_len, dim = x.size()

        # create query from input
        query = self.q_proj(x)
        query = query.view(batch_size, seq_len, self.n_heads, dim // self.n_heads).transpose(1, 2)

        # create keys from template features
        key = self.template_features.view(self.n_symbols, self.n_heads, self.d_model // self.n_heads).transpose(0, 1)
        key = self._repeat_kv(key, batch_size)

        # create values from symbol library
        value = self.symbol_library.view(self.n_symbols, self.n_heads, self.d_model // self.n_heads).transpose(0, 1)
        value = self._repeat_kv(value, batch_size)

        retrieved_symbols = torch.nn.functional.scaled_dot_product_attention(
            query, key, value,
            scale=self.scale, dropout_p=self.dropout, attn_mask=None, is_causal=False)
        retrieved_symbols = retrieved_symbols.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)

        return retrieved_symbols

    def _repeat_kv(self, x, batch_size):
        """
        template_features and symbol_library are of shape (n_heads, n_s, d_s//n_heads).
        repeat for each input and add a batch dimension of size batch_size.
        """
        return x.unsqueeze(0).repeat(batch_size, 1, 1, 1)

class PositionalSymbolRetriever(nn.Module):
    def __init__(self, symbol_dim, max_length, sinusoidal=False):
        """
        Postional Symbol Retriever.

        Learns a library of "symbols".
        Retrieves a symbol for each object based on its position.

        Parameters
        ----------
        symbol_dim : int
            dimension of the symbols.
        max_symbols : int
            maximum number of symbols.
        """

        super().__init__()
        self.symbol_dim = symbol_dim
        self.max_length = max_length
        self.sinusoidal = sinusoidal

        self.symbol_library = nn.Embedding(self.max_length, self.symbol_dim)

        # TODO: implement sinusoidal symbols?

    def forward(self, x):
        device = x.device
        batch_size, seq_len, dim = x.size()

        pos = torch.arange(0, seq_len, dtype=torch.long, device=device)
        retrieved_symbols = self.symbol_library(pos).unsqueeze(0).repeat(batch_size, 1, 1)

        return retrieved_symbols


class PositionRelativeSymbolRetriever(nn.Module):
    def __init__(self, symbol_dim, max_rel_pos):
        """
        Position-Relative Symbol Retriever.

        For i -> j, the symbol s_{ij} encodes the relative position j - i.

        Parameters
        ----------
        symbol_dim : int
            dimension of the symbols.
        max_rel_pos : int
            maximum relative position encoded by symbols.
            Positions exceeding this will be truncated.
        """
        super().__init__()
        self.symbol_dim = symbol_dim
        self.max_rel_pos = max_rel_pos

        self.rel_pos_enc = RelativePositionalEncoding(dim=symbol_dim, max_rel_pos=max_rel_pos)

    def forward(self, x):
        length = x.shape[1]
        return self.rel_pos_enc(length)

class RelationalSymbolicAttention(nn.Module):
    def __init__(self,
            d_model: int,
            rel_n_heads: int,
            symbolic_attn_n_heads: int,
            n_symbols: int,
            nbhd_delta: int,
            causal_nbhd: bool = True,
            include_self: bool = False,
            normalize_rels: bool = True,
            dropout: float = 0.0,
            rel_scale: float = None,
            symbolic_attn_scale: float = None):
        """
        Relational symbolic attention module.

        Retrieves a symbol for each object in the input based on its relationship with its neighborhood.
        First, we compute a local relation vector for each object in the input. This local relation vector
        is then used to retrieve a symbol from the symbol library via symbolic attention.

        Parameters
        ----------
        d_model : int
            Model dimension. this is the dimension of the input and the dimension of the symbols and template features.
        rel_n_heads : int
            Dimensionality of relations computed with neighborhood.
        symbolic_attn_n_heads : int
            Number of symbolic attention heads.
        n_symbols : int
            Number of symbols to learn in the symbol library.
        nbhd_delta : int
            The size of the neighborhood.
        causal_nbhd : bool, optional
            Whether to use causal neighborhood. if causal_nbhd is True, the neighborhood is [i-nbhd_delta, i].
            if causal_nbhd is False, the neighborhood is [i-nbhd_delta, i+nbhd_delta]. Defaults to True.
        include_self : bool, optional
            Whether to include self in the neighborhood. E.g., if False and causal_nbhd, the neighborhood is
            [i-nbhd_delta, i-1]. If False and not causal_nbhd, the neighborhood is [i-nbhd_delta, i-1] U [i+1, i+nbhd_delta].
            Defaults to False.
        normalize_rels : bool, optional
            Whether to normalize relations with softmax across neighborhood. Defaults to True.
        dropout : float, optional
            The dropout rate. Defaults to 0.0.
        rel_scale : float, optional
            The scaling factor when normalizing relations via softmax. If None, it is computed based on model_dim and rel_n_heads.
        symbolic_attn_scale : float, optional
            The scaling factor used in symbolic attention.

        Attributes
        ----------
        symbolic_attention : SymbolicAttention
            The symbolic attention module.
        q_proj : nn.Linear
            Linear layer for projecting the query.
        k_proj : nn.Linear
            Linear layer for projecting the key.
        model_dim_proj : nn.Linear
            Linear layer for projecting the neighborhood relation vector to model_dim.
        """

        super().__init__()

        self.d_model = d_model
        self.rel_n_heads = rel_n_heads
        self.symbolic_attn_n_heads = symbolic_attn_n_heads
        self.n_symbols = n_symbols
        self.nbhd_delta = nbhd_delta
        self.causal_nbhd = causal_nbhd
        self.dropout = dropout
        self.rel_scale = rel_scale if rel_scale is not None else (d_model//rel_n_heads) ** -0.5
        self.symbolic_attn_scale = symbolic_attn_scale
        self.include_self = include_self
        self.normalize_rels = normalize_rels

        self.nbhd_rel_dim = self._compute_nbhd_rel_dim(rel_n_heads, nbhd_delta, causal_nbhd, include_self)

        self.symbolic_attention = SymbolicAttention(d_model, symbolic_attn_n_heads, n_symbols, dropout, symbolic_attn_scale)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.model_dim_proj = nn.Linear(self.nbhd_rel_dim, d_model) # project neighborhood relation vector to model_dim

    def forward(self, x):
        batch_size, seq_len, dim = x.size()

        # compute query and key transformations to compute relations with neighborhood
        query = self.q_proj(x)
        key = self.k_proj(x)

        # reshape to (batch_size, n_heads, n, d_k); i.e., split model_dim into n_heads
        query = query.view(batch_size, seq_len, self.rel_n_heads, self.d_model // self.rel_n_heads).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.rel_n_heads, self.d_model // self.rel_n_heads).transpose(1, 2)

        # compute neighborhood mask
        if self.causal_nbhd:
            neighbor_mask = self.compute_causal_neighbor_mask(seq_len, self.nbhd_delta, self.include_self)
        else:
            neighbor_mask = self.compute_neighbor_mask(seq_len, self.nbhd_delta, self.include_self)

        neighborhood_keys = key[:, :, neighbor_mask] # (batch_size, n_heads, n, Delta, d_k)

        # compute relations with neighborhood
        # einstein summation: R[b,h,i,j] = sum_d Q[b,h,i,d] * nbhd_K[b,h,i,j,d], where nhbd_K[b,h,i,j,d] = K[b,h,i,i-j,d]
        neighbor_rel_tensor = torch.einsum('bhid,bhijd->bhij', query, neighborhood_keys) # (batch_size, n_heads, n, Delta)

        if self.normalize_rels:
            # normalize relations across neigborhood (of size Delta)
            neighbor_rel_tensor = torch.softmax(neighbor_rel_tensor * self.rel_scale, dim=-1)


        # permute dims to shape (batch_size, n, Delta, n_heads)
        neighbor_rel_tensor = neighbor_rel_tensor.permute(0, 2, 3, 1)

        # flatten n_heads dimension to get shape (batch_size, n, nbhd_rel_dim) [e.g., nbhd_rel_dim = Delta * n_heads]
        neighbor_rel_tensor = neighbor_rel_tensor.contiguous().view(batch_size, -1, self.nbhd_rel_dim)

        # project neighborhood relation vector to model_dim
        neighbor_rel_tensor = self.model_dim_proj(neighbor_rel_tensor)

        # compute symbolic attention
        retrieved_symbols = self.symbolic_attention(neighbor_rel_tensor)

        return retrieved_symbols

    def _compute_nbhd_rel_dim(self, rel_n_heads, nbhd_delta, causal_nbhd, include_self):
        '''computes the dimension of the neigborhood relation vector'''

        if causal_nbhd:
            if include_self:
                return rel_n_heads * (nbhd_delta + 1)
            else:
                return rel_n_heads * nbhd_delta
        else:
            if include_self:
                return rel_n_heads * (2 * nbhd_delta + 1)
            else:
                return rel_n_heads * (2 * nbhd_delta)

    @staticmethod
    def compute_neighbor_mask(n, delta, include_self=True):
        '''computes the neighborhood mask for a sequence of length n and neighborhood size delta'''

        sequence = torch.arange(n).unsqueeze(1)
        if include_self:
            neighborhood = torch.arange(-delta, delta + 1).unsqueeze(0)
        else:
            neighborhood = torch.concat([torch.arange(-delta, 0), torch.arange(1, delta + 1)]).unsqueeze(0)

        mask = sequence + neighborhood
        mask = mask.clamp(0, n - 1)
        return mask

    @staticmethod
    def compute_causal_neighbor_mask(n, delta, include_self=False):
        '''computes the causal neighborhood mask for a sequence of length n and neighborhood size delta'''

        sequence = torch.arange(n).unsqueeze(1)
        if include_self:
            neighborhood = torch.arange(delta + 1).unsqueeze(0)
        else:
            neighborhood = torch.arange(1, delta + 1).unsqueeze(0)

        mask = sequence - neighborhood
        mask = mask.clamp(0, n - 1)
        return mask
