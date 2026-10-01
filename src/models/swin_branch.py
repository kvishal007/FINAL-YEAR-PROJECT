"""
Swin Transformer branch (global feature extractor) for the DeFaX-style model.

This module only EXTRACTS FEATURES. The ImageNet classification head is
removed, so the branch never outputs 1000-class ImageNet predictions.

Backbone: torchvision Swin-T (smallest standard Swin, ~28M parameters),
chosen to fit a 4 GB GPU.

Input:
    images: [batch_size, 3, 224, 224]
        RGB tensors normalized with ImageNet mean/std
        (as done in src/data/transforms.py)

Outputs:
    forward(images)          -> [batch_size, 768]
        One global feature vector per image (average of all tokens).

    forward_tokens(images)   -> [batch_size, 49, 768]
        49 spatial tokens (a 7x7 grid over the image), 768 features each.
        This is the form the cross-attention module will use later.
"""

import torch
from torch import nn
from torchvision.models import Swin_T_Weights, swin_t

SWIN_FEATURE_DIM = 768   # channels in the last Swin-T stage
SWIN_NUM_TOKENS = 49     # 7 x 7 grid for a 224 x 224 input


class SwinBranch(nn.Module):
    def __init__(self, pretrained=True, freeze=False):
        """
        pretrained: load ImageNet weights (downloads ~108 MB on first use).
                    Use False for offline tests with random weights.
        freeze:     stop gradient updates for the backbone. Saves GPU memory
                    during training because no gradients are stored for it.
        """
        super().__init__()

        weights = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = swin_t(weights=weights)

        # Keep only the feature layers; the ImageNet head is discarded
        self.features = backbone.features  # patch embedding + 4 Swin stages
        self.norm = backbone.norm          # final LayerNorm
        self.feature_dim = SWIN_FEATURE_DIM

        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def forward_tokens(self, images):
        """[B, 3, 224, 224] -> [B, 49, 768] spatial feature tokens."""
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(
                f"Expected input [batch_size, 3, H, W], got {list(images.shape)}"
            )

        x = self.features(images)  # [B, 7, 7, 768] (channels last)
        x = self.norm(x)
        return x.flatten(1, 2)     # [B, 49, 768]

    def forward(self, images):
        """[B, 3, 224, 224] -> [B, 768] global feature vector."""
        return self.forward_tokens(images).mean(dim=1)
