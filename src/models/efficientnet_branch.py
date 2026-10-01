"""
EfficientNet branch (local feature extractor) for the DeFaX-style model.

This module only EXTRACTS FEATURES. The ImageNet classifier is removed, so
the branch never outputs 1000-class ImageNet predictions.

Backbone: torchvision EfficientNet-B0 (~4M parameters without the classifier),
designed for 224 x 224 input and light enough for a 4 GB GPU.

Input:
    images: [batch_size, 3, 224, 224]
        RGB tensors normalized with ImageNet mean/std
        (as done in src/data/transforms.py)

Outputs:
    forward(images)              -> [batch_size, 1280]
        One global feature vector per image (average over the 7x7 map).

    forward_feature_map(images)  -> [batch_size, 1280, 7, 7]
        Spatial feature map before global pooling (channels first).

    forward_tokens(images)       -> [batch_size, 49, 1280]
        The same 7x7 map as 49 tokens, matching the token layout of
        SwinBranch.forward_tokens() ([batch_size, 49, 768]) so the two
        branches can be fused by cross-attention later.
"""

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

EFFICIENTNET_FEATURE_DIM = 1280  # channels in the last EfficientNet-B0 layer
EFFICIENTNET_NUM_TOKENS = 49     # 7 x 7 grid for a 224 x 224 input


class EfficientNetBranch(nn.Module):
    def __init__(self, pretrained=True, freeze=False):
        """
        pretrained: load ImageNet weights (downloads ~21 MB on first use).
                    Use False for offline tests with random weights.
        freeze:     stop gradient updates for the backbone and keep its
                    BatchNorm statistics fixed. Saves GPU memory in training.
        """
        super().__init__()

        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b0(weights=weights)

        # Keep only the convolutional feature layers; the ImageNet
        # classifier (Dropout + Linear 1280 -> 1000) is discarded
        self.features = backbone.features
        self.feature_dim = EFFICIENTNET_FEATURE_DIM
        self.frozen = freeze

        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        # BatchNorm layers update their running statistics in train mode even
        # when requires_grad is False, so a frozen backbone must stay in eval
        super().train(mode)
        if self.frozen:
            self.features.eval()
        return self

    def forward_feature_map(self, images):
        """[B, 3, 224, 224] -> [B, 1280, 7, 7] spatial feature map."""
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(
                f"Expected input [batch_size, 3, H, W], got {list(images.shape)}"
            )
        return self.features(images)

    def forward_tokens(self, images):
        """[B, 3, 224, 224] -> [B, 49, 1280] spatial feature tokens."""
        feature_map = self.forward_feature_map(images)  # [B, 1280, 7, 7]
        return feature_map.flatten(2).transpose(1, 2)   # [B, 49, 1280]

    def forward(self, images):
        """[B, 3, 224, 224] -> [B, 1280] global feature vector."""
        return self.forward_feature_map(images).mean(dim=(2, 3))
