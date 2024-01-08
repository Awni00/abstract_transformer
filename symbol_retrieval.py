import torch
import torch.nn as nn

class SymbolicAttention(nn.Module):
    def __init__(self, model_dim, n_heads, num_symbols, dropout=0.0, scale=None):
        """
        Symbolic Attention.

        Learns a library of "symbols" and corresponding template features.
        For a given input, retrieves a symbol from the symbol library via attention.

        Parameters
        ----------
        model_dim : int
            model dimension. this is the dimension of the input and the dimension of the symbols and template features.
        n_heads : int
            number of heads in symbolic attention.
        num_symbols : int
            number of symbols in the symbol library.
        dropout : float, optional
            dropout probability, by default 0.0
        scale : float, optional
            scaling factor in scaled_dot_product_attention, by default None
        """

        super().__init__()
        self.model_dim = model_dim
        self.n_heads = n_heads
        self.num_symbols = num_symbols
        self.dropout = dropout
        self.scale = scale

        self.q_proj = nn.Linear(self.model_dim, self.model_dim)
        self.template_features = nn.Parameter(torch.empty(self.num_symbols, self.model_dim))
        self.symbol_library = nn.Parameter(torch.empty(self.num_symbols, self.model_dim))

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
        key = self.template_features.view(self.num_symbols, self.n_heads, self.model_dim // self.n_heads).transpose(0, 1)
        key = self._repeat_kv(key, batch_size)

        # create values from symbol library
        value = self.symbol_library.view(self.num_symbols, self.n_heads, self.model_dim // self.n_heads).transpose(0, 1)
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
