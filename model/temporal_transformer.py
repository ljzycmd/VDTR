""" ************************************************
* fileName: temporal_attn.py
* desc: The temporal fusion stategies in deblurring transformer.
* author: mingdeng_cao
* date: 2021/07/07 15:59
* last revised: None
************************************************ """


import torch
import torch.nn as nn
import torch.nn.functional as F

from einops.layers.torch import Rearrange
from einops import rearrange, repeat

from .local_transformer import FFN


class TemporalTransformer(nn.Module):
    def __init__(
        self,
        embedding_dim=128,
        num_frames=5,
        patch_size=4,
        num_heads=3,
        dropout=0.0,
        shift_size=0,
    ):
        super().__init__()
        self.embedding = embedding_dim
        self.num_frames = num_frames
        self.patch_size = patch_size
        self.shift_size = shift_size
        self.to_patches = nn.Sequential(
            Rearrange(
                "b n c (hp p1) (wp p2) -> (n p1 p2) (b hp wp)  c",
                p1=patch_size,
                p2=patch_size,
            )
        )

        # multi-head attention
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.attn = nn.MultiheadAttention(embedding_dim, num_heads, dropout)
        # ffn
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.ffn = FFN(embedding_dim, embedding_dim * 4, dropout)

        # temporal 1D pos encoding learnable for multi-frame fusion
        self.row_embed = nn.Parameter(
            torch.Tensor(self.patch_size, self.embedding // 3)
        )
        self.col_embed = nn.Parameter(
            torch.Tensor(self.patch_size, self.embedding // 3)
        )
        self.depth_embed = nn.Parameter(
            torch.Tensor(self.num_frames, self.embedding - self.embedding // 3 * 2)
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed)
        nn.init.uniform_(self.col_embed)
        nn.init.uniform_(self.depth_embed)

    def get_learnable_pos(self):
        pos = (
            torch.cat(
                [
                    self.row_embed.unsqueeze(0)
                    .unsqueeze(2)
                    .repeat(self.num_frames, 1, self.patch_size, 1),
                    self.col_embed.unsqueeze(0)
                    .unsqueeze(0)
                    .repeat(self.num_frames, self.patch_size, 1, 1),
                    self.depth_embed.unsqueeze(1)
                    .unsqueeze(1)
                    .repeat(1, self.patch_size, self.patch_size, 1),
                ],
                dim=-1,
            )
            .flatten(0, 2)
            .unsqueeze(1)
        )  # (n*p*p, 1, c)

        return pos  # (n*p*p, 1, c)

    def forward(self, x, ref_idx=None):
        """
        Args:
            x: (b, n, c, h, w), the input_feature maps
            ref_idx: the reference frame index
        """
        assert x.dim() == 5, "Input feature map should have 5 dims!"
        b, n, c, h, w = x.shape

        # shift the non-overlapping patches
        if self.shift_size > 0:
            x = torch.roll(
                x, shifts=(self.shift_size, self.shift_size), dims=(3, 4)
            )

        # padding the feature maps
        padding_h = (self.patch_size - h % self.patch_size) % self.patch_size
        padding_w = (self.patch_size - w % self.patch_size) % self.patch_size

        x = F.pad(x, (0, padding_w, 0, padding_h))  # (l, r, t, b)
        h_paded, w_paded = x.shape[-2:]

        # split into patches
        x_patches = self.to_patches(x)  # (n*p*p, b*h/p*w/p, c)
        if ref_idx is not None:
            residual = x_patches[ref_idx*self.patch_size*self.patch_size:(ref_idx+1)*self.patch_size*self.patch_size]
        else:
            residual = x_patches
        # multi-head inner patch attention
        x_patches = self.norm1(x_patches)

        # adding the positional encoding
        x_patches += self.get_learnable_pos()

        # attention aross multiple frames (n*p*p, b*h/p*w/p, c)
        if ref_idx is not None:
            query = x_patches[ref_idx*self.patch_size*self.patch_size:(ref_idx+1)*self.patch_size*self.patch_size]
        else:
            query = x_patches
        x_attn, _ = self.attn(query=query,
                              key=x_patches, value=x_patches)

        # residual
        x = x_attn + residual

        # feedforward network
        x = x + self.ffn(self.norm2(x))

        # patches to feature map
        if ref_idx is None:
            x = rearrange(
                x,
                "(n p1 p2) (b hp wp) c -> b n c (hp p1) (wp p2)",
                n=n,
                p1=self.patch_size,
                p2=self.patch_size,
                hp=h_paded // self.patch_size,
                wp=w_paded // self.patch_size,
            )
        else:
            x = rearrange(
                x,
                "(p1 p2) (b hp wp) c -> b c (hp p1) (wp p2)",
                p1=self.patch_size,
                p2=self.patch_size,
                hp=h_paded // self.patch_size,
                wp=w_paded // self.patch_size,
            )

        if padding_h > 0 or padding_w > 0:
            x = x[..., :h, :w]

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -
                           self.shift_size), dims=(-2, -1))

        return x


class TemporalFusion(nn.Module):
    def __init__(
        self,
        embedding_dim=96,
        num_frames=5,
        patch_size=4,
        num_heads=3,
        dropout=0.0,
        two_layer=True,
    ):
        super().__init__()
        self.embedding = embedding_dim
        self.num_frames = num_frames
        self.patch_size = patch_size
        self.two_layer = two_layer

        self.temp_transformer = TemporalTransformer(
            embedding_dim,
            num_frames,
            patch_size,
            num_heads,
            dropout,
            0
        )
        if self.two_layer:
            self.temp_transformer2 = TemporalTransformer(
                embedding_dim,
                num_frames,
                patch_size,
                num_heads,
                dropout,
                patch_size // 2
            )

    def forward(self, x):
        """
        Args:
            x: (b, n, c, h, w), the input_feature maps
        """
        assert x.dim() == 5, "Input feature map should have 5 dims!"
        b, n, c, h, w = x.shape
        if self.two_layer:
            x = self.temp_transformer2(x)
        out = self.temp_transformer(x, ref_idx=n//2)

        return out
