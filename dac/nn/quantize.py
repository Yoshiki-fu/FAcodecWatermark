from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils import weight_norm

from dac.nn.layers import WNConv1d


class VectorQuantize(nn.Module):
    """
    Implementation of VQ similar to Karpathy's repo:
    https://github.com/karpathy/deep-vector-quantization
    Additionally uses following tricks from Improved VQGAN
    (https://arxiv.org/pdf/2110.04627.pdf):
        1. Factorized codes: Perform nearest neighbor lookup in low-dimensional space
            for improved codebook usage
        2. l2-normalized codes: Converts euclidean distance to cosine similarity which
            improves training stability
    """

    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.codebook_size = codebook_size      # 1024
        self.codebook_dim = codebook_dim        # 8

        self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1)     # 1024->1024
        self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1)        # 1024->1024
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def forward(self, z):
        """Quantized the input tensor using a fixed codebook and returns
        the corresponding codebook vectors

        Parameters
        ----------
        z : Tensor[B x D x T]

        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input
        Tensor[1]
            Commitment loss to train encoder to predict vectors closer to codebook
            entries
        Tensor[1]
            Codebook loss to update the codebook
        Tensor[B x T]
            Codebook indices (quantized discrete representation of input)
        Tensor[B x D x T]
            Projected latents (continuous representation of input before quantization)
        """

        # Factorized codes (ViT-VQGAN) Project input into low-dimensional space
        z_e = self.in_proj(z)  # z_e : (B x D x T)
        z_q, indices = self.decode_latents(z_e)

        commitment_loss = F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2])      # エンコーダを更新するためのロス
        codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2])        # コードブックを更新するためのロス

        z_q = (
            z_e + (z_q - z_e).detach()
        )  # noop in forward pass, straight-through gradient estimator in backward pass     エンコーダへ勾配を流すためのトリック

        z_q = self.out_proj(z_q)

        return z_q, commitment_loss, codebook_loss, indices, z_e

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id):
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(self, latents):
        encodings = rearrange(latents, "b d t -> (b t) d")
        codebook = self.codebook.weight  # codebook: (N=1024 x D=8)

        # L2 normalize encodings and codebook (ViT-VQGAN)
        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)

        # Compute euclidean distance with codebook
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
        z_q = self.decode_code(indices)
        return z_q, indices


class ResidualVectorQuantize(nn.Module):
    """
    Introduced in SoundStream: An end2end neural audio codec
    https://arxiv.org/abs/2107.03312
    """

    def __init__(
        self,
        input_dim: int = 512,       # 1024
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: float = 0.0,
    ):
        super().__init__()
        if isinstance(codebook_dim, int):
            codebook_dim = [codebook_dim for _ in range(n_codebooks)]

        self.n_codebooks = n_codebooks
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size

        self.quantizers = nn.ModuleList(
            [VectorQuantize(input_dim, codebook_size, codebook_dim[i]) for i in range(n_codebooks)]
        )
        self.quantizer_dropout = quantizer_dropout

    def forward(self, z, n_quantizers: int = None):
        """Quantized the input tensor using a fixed set of `n` codebooks and returns
        the corresponding codebook vectors
        Parameters
        ----------
        z : Tensor[B x D x T]
        n_quantizers : int, optional
            No. of quantizers to use
            (n_quantizers < self.n_codebooks ex: for quantizer dropout)
            Note: if `self.quantizer_dropout` is True, this argument is ignored
                when in training mode, and a random number of quantizers is used.
        Returns
        -------
        dict
            A dictionary with the following keys:

            "z" : Tensor[B x D x T]
                Quantized continuous representation of input
            "codes" : Tensor[B x N x T]
                Codebook indices for each codebook
                (quantized discrete representation of input)
            "latents" : Tensor[B x N*D x T]
                Projected latents (continuous representation of input before quantization)
            "vq/commitment_loss" : Tensor[1]
                Commitment loss to train encoder to predict vectors closer to codebook
                entries
            "vq/codebook_loss" : Tensor[1]
                Codebook loss to update the codebook
        """
        z_q = 0
        residual = z
        commitment_loss = 0
        codebook_loss = 0

        codebook_indices = []
        latents = []
        out_quantized = []

        if n_quantizers is None:
            n_quantizers = self.n_codebooks

        # ランダムな量子化数ドロップアウト
        """if self.training:
            n_quantizers = torch.ones((z.shape[0],)) * self.n_codebooks + 1
            dropout = torch.randint(1, self.n_codebooks + 1, (z.shape[0],))
            n_dropout = int(z.shape[0] * self.quantizer_dropout)
            n_quantizers[:n_dropout] = dropout[:n_dropout]
            n_quantizers = n_quantizers.to(z.device)"""

        for i, quantizer in enumerate(self.quantizers):
            if self.training is False and i >= n_quantizers:
                break

            z_q_i, commitment_loss_i, codebook_loss_i, indices_i, z_e_i = quantizer(
                residual
            )

            # Create mask to apply quantizer dropout
            """mask = (
                torch.full((z.shape[0],), fill_value=i, device=z.device) < n_quantizers
            )"""
            z_q = z_q + z_q_i       #* mask[:, None, None]
            residual = residual - z_q_i

            # Sum losses
            """commitment_loss += (commitment_loss_i * mask).mean()
            codebook_loss += (codebook_loss_i * mask).mean()"""
            commitment_loss += commitment_loss_i.mean()
            codebook_loss += codebook_loss_i.mean()

            codebook_indices.append(indices_i)
            latents.append(z_e_i)
            out_quantized.append(z_q_i)

        codes = torch.stack(codebook_indices, dim=1)
        latents = torch.cat(latents, dim=1)

        return z_q, codes, latents, commitment_loss, codebook_loss, out_quantized     # Z_q: 量子化器の出力, codes:各量子化器の何番目のコードブックを使ったか, latents:エンコーダの出力, commitment_loss:エンコーダを更新するためのロス, codebook_loss:コードブックを更新するためのロス

    def from_codes(self, codes: torch.Tensor):
        """Given the quantized codes, reconstruct the continuous representation
        Parameters
        ----------
        codes : Tensor[B x N x T]
            Quantized discrete representation of input
        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input
        """
        z_q = 0.0
        z_p = []
        n_codebooks = codes.shape[1]
        for i in range(n_codebooks):
            z_p_i = self.quantizers[i].decode_code(codes[:, i, :])
            z_p.append(z_p_i)

            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i
        return z_q, torch.cat(z_p, dim=1), codes

    def from_latents(self, latents: torch.Tensor):
        """Given the unquantized latents, reconstruct the
        continuous representation after quantization.

        Parameters
        ----------
        latents : Tensor[B x N x T]
            Continuous representation of input after projection

        Returns
        -------
        Tensor[B x D x T]
            Quantized representation of full-projected space
        Tensor[B x D x T]
            Quantized representation of latent space
        """
        z_q = 0
        z_p = []
        codes = []
        dims = np.cumsum([0] + [q.codebook_dim for q in self.quantizers])

        n_codebooks = np.where(dims <= latents.shape[1])[0].max(axis=0, keepdims=True)[
            0
        ]
        for i in range(n_codebooks):
            j, k = dims[i], dims[i + 1]
            z_p_i, codes_i = self.quantizers[i].decode_latents(latents[:, j:k, :])
            z_p.append(z_p_i)
            codes.append(codes_i)

            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i

        return z_q, torch.cat(z_p, dim=1), torch.stack(codes, dim=1)
    
    def get_pre_vq_continuous_feature(self, z: torch.Tensor, layer_idx: int = 0) -> torch.Tensor:
        """
        指定した量子化層の、in_proj（射影層）を通過した直後の連続値特徴量を取得します。
        
        Parameters
        ----------
        z : Tensor[B x D x T]
            RVQモジュールへの入力テンソル（広い潜在特徴量）
        layer_idx : int, optional
            取得したい量子化層のインデックス。デフォルトは 0 (1本目のベクトル用)。

        Returns
        -------
        Tensor[B x codebook_dim x T]
            量子化される直前の、低次元（または同次元）に射影された連続値特徴量 (z_e)
        """
        if layer_idx >= len(self.quantizers):
            raise ValueError(f"layer_idx {layer_idx} is out of bounds for {len(self.quantizers)} quantizers.")
        
        # 指定された層（通常は0番目）の in_proj を使って射影する
        # ※ self.quantizers[layer_idx] は VectorQuantize クラスのインスタンス
        # 1. 1024 -> 8 に圧縮（ここで声質などの不要情報が削ぎ落とされる）
        z_e = self.quantizers[layer_idx].in_proj(z)
        
        # 2. 8 -> 1024 に復元（量子化はスキップし、連続値のまま広げる）
        z_filtered = self.quantizers[layer_idx].out_proj(z_e)
        
        return z_filtered



if __name__ == "__main__":
    rvq = ResidualVectorQuantize(quantizer_dropout=True)
    x = torch.randn(16, 512, 80)
    y = rvq(x)
    print(y["latents"].shape)
