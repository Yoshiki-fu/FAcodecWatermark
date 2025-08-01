import torch
import torch.nn as nn
import torch.nn.functional as F

class FCBlock(nn.Module):
    """ Fully Connected Block """

    def __init__(self, in_features, out_features, activation=None, bias=False, dropout=None, spectral_norm=False):
        super(FCBlock, self).__init__()
        self.fc_layer = nn.Sequential()
        self.fc_layer.add_module(
            "fc_layer",
            LinearNorm(
                in_features,
                out_features,
                bias,
                spectral_norm,
            ),
        )
        if activation is not None:
            self.fc_layer.add_module("activ", activation)
        self.dropout = dropout

    def forward(self, x):
        x = self.fc_layer(x)
        if self.dropout is not None:
            x = F.dropout(x, self.dropout, self.training)
        return x

class LinearNorm(nn.Module):
    """ LinearNorm Projection """

    def __init__(self, in_features, out_features, bias=False, spectral_norm=False):
        super(LinearNorm, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

        nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            nn.init.constant_(self.linear.bias, 0.0)
        if spectral_norm:
            self.linear = nn.utils.spectral_norm(self.linear)

    def forward(self, x):
        x = self.linear(x)
        return x

class WMEmbedder(nn.Module):
    """
    A class that takes a secret message, processes it into chunk embeddings
    (as a small sequence), and uses a TransformerDecoder to do cross-attention
    between the original hidden (target) and the watermark tokens (memory).
    """

    def __init__(
        self,
        nbits: int,  # 16 total bits in the secret message
        input_dim: int,  # 1024 the input dimension (e.g. audio feature dimension)
        nchunk_size: int,       # 4
        hidden_dim: int = 256,
        num_heads: int = 1,
        num_layers: int = 4,
    ):
        super().__init__()
        self.nchunk_size = nchunk_size
        assert nbits % nchunk_size == 0, "nbits must be a multiple of nchunk_size!"
        self.nbits = nbits
        self.nchunks = nbits // nchunk_size  # how many chunks

        # Each chunk (0..2^nchunk_size - 1) maps to an embedding of size [hidden_dim]
        self.msg_embeddings = nn.ModuleList(
            nn.Embedding(2**nchunk_size, hidden_dim) for _ in range(self.nchunks)
        )

        # Linear to project [input_dim] -> [hidden_dim]
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # TransformerDecoder for cross-attention
        # d_model=hidden_dim, so the decoder expects [b, seq_len, hidden_dim] as tgt
        # and [b, memory_len, hidden_dim] as memory
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=2 * hidden_dim,
            activation="gelu",
            batch_first=True,  # so shape is [batch, seq, feature]
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

        # Project [hidden_dim] -> [input_dim]
        # self.output_projection1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, input_dim)

    def forward(self, hidden: torch.Tensor, msg: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden: [batch, input_dim, seq_len]
            msg: [batch, nbits]
        Returns:
            A tensor [batch, input_dim, seq_len] with watermark injected.
        """
        b, in_dim, seq_len = hidden.shape

        # 1) Project input features to [b, seq_len, hidden_dim]     [b, sqe_len, 1024] -> [b, seq_len, 256]
        hidden_projected = self.input_projection(
            hidden.permute(0, 2, 1)
        )  # => [b, seq_len, hidden_dim]

        # 2) Convert the msg bits into a sequence of chunk embeddings
        #    We keep each chunk as one token => [b, nchunks, hidden_dim]
        chunk_emb_list = []
        for i in range(self.nchunks):
            # msg[:, i*nchunk_size : (i+1)*nchunk_size] => shape [b, nchunk_size]
            chunk_bits = msg[:, i * self.nchunk_size : (i + 1) * self.nchunk_size]
            chunk_val = torch.zeros_like(chunk_bits[:, 0])  # shape [b]
            for bit_idx in range(self.nchunk_size):
                # shift bits
                chunk_val += chunk_bits[:, bit_idx] << bit_idx

            # embedding => [b, hidden_dim]
            chunk_emb = self.msg_embeddings[i](chunk_val)
            chunk_emb_list.append(chunk_emb.unsqueeze(1))  # => [b,1,hidden_dim]

        # Concat => [b, nchunks, hidden_dim]
        chunk_emb_seq = torch.cat(chunk_emb_list, dim=1)  # [b, nchunks, hidden_dim]

        # 3) Use chunk_emb_seq as memory, hidden_projected as target for TransformerDecoder
        #
        # TransformerDecoder forward signature:
        #   transformer_decoder(tgt, memory, ...)
        #   => [b, seq_len, hidden_dim]
        x_decoded = self.transformer_decoder(
            tgt=hidden_projected,  # [b, seq_len, hidden_dim]
            memory=chunk_emb_seq,  # [b, nchunks, hidden_dim]
        )

        # 4) Project back to input_dim => [b, seq_len, input_dim]
        x_output = self.output_projection(x_decoded)

        # 5) permute back to [b, input_dim, seq_len]
        x_output = x_output.permute(0, 2, 1)  # => [b, input_dim, seq_len]

        # 6) (Optional) Residual with original hidden
        x_output = x_output + hidden

        return x_output
    

def random_message(nbits: int, batch_size: int) -> torch.Tensor:
    """Return random message as 0/1 tensor."""
    if nbits == 0:
        return torch.tensor([])
    return torch.randint(0, 2, (batch_size, nbits))