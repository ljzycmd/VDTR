""" ************************************************
* fileName: trans_unet.py
* author: mingdeng_cao
* date: 2021/07/07 15:42
* last revised: None
************************************************ """


import torch
import torch.nn as nn
import torch.nn.functional as F

from .local_transformer import Upsample, Downsample, LocalAttnLayer


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        inner_channels,
        down_times=4,
        patch_size=4,
        num_layers=12,
        num_heads=3,
        dropout=0.0,
        ffn_dim=None,
        attn_type="LocalAttnLayer",
        *args,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.inner_channels = inner_channels
        self.down_times = down_times
        self.patch_size = patch_size
        if down_times > 1:
            self.downsample = Downsample(in_channels, inner_channels, down_times)

        self.attn_layers = nn.Sequential(
            *[
                LocalAttnLayer(
                    self.inner_channels,
                    self.patch_size,
                    num_heads,
                    dropout,
                    shift_size=0 if i % 2 == 0 else self.patch_size // 2,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        """
        feats = x
        if self.down_times > 1:
            feats = self.downsample(feats)  # (b, c, h/d, w/d)
        return self.attn_layers(feats)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        inner_channels,
        up_times=4,
        patch_size=4,
        num_layers=12,
        num_heads=3,
        dropout=0.0,
        ffn_dim=None,
        fusion="add",
        *args,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.inner_channels = inner_channels
        self.up_times = up_times
        self.patch_size = patch_size
        if self.up_times > 1:
            self.upsample = Upsample(in_channels, inner_channels, up_times)
        self.fusion = fusion
        if fusion == "concat":
            self.fusion_linear = nn.Conv2d(inner_channels*2, inner_channels, 1, 1, 0)
        self.attn_layers = nn.Sequential(
            *[
                LocalAttnLayer(
                    self.inner_channels,
                    self.patch_size,  # if (i % 2 == 0) else 6,
                    num_heads,
                    dropout,
                    shift_size=0 if i % 2 == 0 else self.patch_size // 2,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, x, encoded_feats=None):
        """
        Args:
            x: (b, c, h, w)
        """
        feats = self.upsample(x)  # (b, c, h*p, w*p)
        if encoded_feats is not None:
            if self.fusion == "add":
                feats += encoded_feats
            elif self.fusion == "concat":
                feats = self.fusion_linear(torch.cat([encoded_feats, feats], dim=1))
        return self.attn_layers(feats)


class TransUnet(nn.Module):
    def __init__(
        self,
        in_channels,
        patch_size=4,
        num_encoder_layers=[2, 2, 2],
        num_decoder_layers=[2, 2],
        num_heads=3,
        dropout=0.0,
        decoder_fusion="add",
        ffn_dim=None,
        return_ms=False,
        ms_fuse=False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.inner_channels = None
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.return_ms = return_ms
        self.ms_fuse = ms_fuse
        if ms_fuse:
            self.ms_fusion_linear = nn.Conv2d(3 * in_channels, in_channels, 1, 1, 0)

        self.num_stages = len(num_encoder_layers)

        self.encoder_stages = nn.ModuleList()
        self.decoder_stages = nn.ModuleList()

        for i in range(self.num_stages):
            if i == 0:
                self.encoder_stages.append(
                    EncoderBlock(
                        # in_channels=in_channels * 2 ** i,
                        # inner_channels=in_channels * 2 ** (i + 1),
                        in_channels=in_channels,
                        inner_channels=in_channels,
                        down_times=1,
                        patch_size=patch_size,
                        num_layers=num_encoder_layers[i],
                        num_heads=num_heads,
                        dropout=dropout,
                        *args,
                        **kwargs,
                    )
                )
            else:
                self.encoder_stages.append(
                    EncoderBlock(
                        # in_channels=in_channels * 2 ** i,
                        # inner_channels=in_channels * 2 ** (i + 1),
                        in_channels=in_channels,
                        inner_channels=in_channels,
                        down_times=2,
                        patch_size=patch_size,
                        num_layers=num_encoder_layers[i],
                        num_heads=num_heads,
                        dropout=dropout,
                        *args,
                        **kwargs,
                    )
                )

        for i in range(self.num_stages - 1):
            self.decoder_stages.append(
                DecoderBlock(
                    # in_channels=in_channels * 2 ** (self.num_stages - i),
                    # inner_channels=in_channels *
                    # 2 ** (self.num_stages - i - 1),
                    in_channels=in_channels,
                    inner_channels=in_channels,
                    up_times=2,
                    patch_size=4,
                    num_layers=num_decoder_layers[i],
                    num_heads=num_heads,
                    dropout=dropout,
                    fusion=decoder_fusion,
                    *args,
                    **kwargs,
                )
            )

    def forward(self, x):
        assert x.dim() == 4, "Input tensor should be in 4 dims!"
        encoder_out = []
        for i in range(self.num_stages):
            if i == 0:
                encoder_out.append(self.encoder_stages[i](x))
            else:
                encoder_out.append(self.encoder_stages[i](encoder_out[-1]))

        decoder_out = [encoder_out[-1]]
        for i in range(self.num_stages - 1):
            decoder_out.append(self.decoder_stages[i](
                decoder_out[-1], encoder_out[-i - 2]))

        if self.ms_fuse:
            for i in range(self.num_stages):
                decoder_out[i] = F.interpolate(decoder_out[i], x.shape[-2:], mode="bilinear")
            decoder_out = torch.cat(decoder_out, dim=1)
            decoder_out = self.ms_fusion_linear(decoder_out)
            return decoder_out

        if self.return_ms:
            return decoder_out

        return decoder_out[-1]
