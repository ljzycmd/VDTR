""" ************************************************
* fileName: dbtrans_v2.py
* desc: video deblurring with transformer version 2
* author: mingdeng_cao
* date: 2021/04/09 16:41
* last revised: 2021.7.29 add pwcnet for multi-frame
                          alignment. 
************************************************ """

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from .local_transformer import Upsample, Downsample
from .trans_unet import EncoderBlock, DecoderBlock, TransUnet

from .temporal_transformer import TemporalFusion

from simdeblur.model.build import BACKBONE_REGISTRY


class BasicBlock(nn.Module):
    def __init__(self, in_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)

    def forward(self, x):
        residual = x

        out = self.relu(self.conv1(x))
        out = self.conv2(out)

        out = out + residual
        out = self.relu(out)

        return out


@BACKBONE_REGISTRY.register()
class VDTR(nn.Module):
    def __init__(
        self,
        in_channels=3,
        inner_channels=256,
        num_frames=5,
        patch_size=4,
        cnn_patch_embedding=False,
        patch_embedding_size=4,
        temporal_patchsize=4,
        temporal_two_layer=True,
        num_layer_rec=20,
        num_heads=8,
        dropout=0.0,
        ffn_dim=None,
        ms_fuse=False
    ):
        super().__init__()
        self.num_frames = num_frames
        self.down_times = patch_embedding_size

        if cnn_patch_embedding:
            self.img2feats = nn.Sequential(
                nn.Conv2d(in_channels, inner_channels//2, 3, 2, 1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(inner_channels//2, inner_channels, 3, 2, 1),
                nn.LeakyReLU(0.1)
            )
        else:
            self.img2feats = Downsample(in_channels, inner_channels, patch_embedding_size)

        self.feature_encoder = TransUnet(
                inner_channels,
                patch_size=4,
                num_encoder_layers=[1, 1, 1],
                num_decoder_layers=[1, 1],
                num_heads=num_heads,
                dropout=dropout,
                ffn_dim=ffn_dim,
                return_ms=False,
                ms_fuse=ms_fuse
            )

        self.temporal_fusion = TemporalFusion(
            inner_channels,
            num_frames,
            temporal_patchsize,
            num_heads,
            dropout,
            temporal_two_layer
        )

        self.reconstructor = nn.Sequential(
            EncoderBlock(
                inner_channels,
                inner_channels,
                down_times=1,
                patch_size=patch_size,
                num_layers=num_layer_rec,
                num_heads=num_heads,
                dropout=dropout,
                ffn_dim=ffn_dim,
            )
        ) 

        self.upsample = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels*4, 1, 1, 0),
            nn.PixelShuffle(2),
            nn.Conv2d(inner_channels, inner_channels*4, 1, 1, 0),
            nn.PixelShuffle(2)
        )

        self.out_proj = nn.Sequential(
            nn.Conv2d(inner_channels, 3, 1, 1, 0))

    def forward(self, x):
        assert x.dim() == 5, "Input tensor should be in 5 dims!"
        B, N, C, H, W = x.shape
        # image to featurs
        feats = self.img2feats(x.flatten(0, 1))

        # single frame feature extraction
        feats = self.feature_encoder(feats)
        feats = feats.reshape(
            B, N, -1, H // self.down_times, W // self.down_times)

        out = self.temporal_fusion(feats)

        out = self.reconstructor(out)
        out = self.upsample(out)
        out = self.out_proj(out)

        # global residual learning
        out += x[:, self.num_frames // 2]

        return out


if __name__ == "__main__":
    pass
