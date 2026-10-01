"""
Complete DeFaX-style model: connects the four verified modules.

    images                      [B, 3, 224, 224]
      ├─ SwinBranch.forward_tokens()          -> swin_tokens  [B, 49, 768]
      └─ EfficientNetBranch.forward_tokens()  -> eff_tokens   [B, 49, 1280]
    CrossAttentionFusion(swin -> Q, eff -> K/V) -> fused_tokens [B, 49, 256]
    mean over tokens (same as forward_fused)  -> fused        [B, 256]
    Classifier                                -> logits       [B, 2]

Labels: index 0 = FAKE, index 1 = REAL.

The model returns raw logits. Use logits.softmax(dim=1) for probabilities.
"""

from torch import nn

from src.models.classifier import Classifier
from src.models.cross_attention import CrossAttentionFusion
from src.models.efficientnet_branch import EFFICIENTNET_FEATURE_DIM, EfficientNetBranch
from src.models.swin_branch import SWIN_FEATURE_DIM, SwinBranch


class DeFaX(nn.Module):
    def __init__(
        self,
        *,
        pretrained,
        freeze_backbones=False,
        attention_dim=256,
        num_heads=8,
        attention_dropout=0.1,
        hidden_dim=128,
        classifier_dropout=0.3,
        num_classes=2,
    ):
        """
        pretrained:        REQUIRED. True loads ImageNet weights for both
                           backbones (downloads ~130 MB on first use);
                           False uses random weights (offline tests).
        freeze_backbones:  freeze Swin and EfficientNet, so only the fusion
                           and classifier learn (much less GPU memory).
        Other arguments are passed to CrossAttentionFusion and Classifier.
        """
        super().__init__()

        self.swin = SwinBranch(pretrained=pretrained, freeze=freeze_backbones)
        self.efficientnet = EfficientNetBranch(pretrained=pretrained, freeze=freeze_backbones)
        self.fusion = CrossAttentionFusion(
            swin_dim=SWIN_FEATURE_DIM,
            efficientnet_dim=EFFICIENTNET_FEATURE_DIM,
            attention_dim=attention_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
        )
        self.classifier = Classifier(
            input_dim=attention_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=classifier_dropout,
        )

    def forward(self, images, return_features=False, return_attention=False):
        """
        images: [B, 3, 224, 224]

        Default: returns logits [B, 2].

        return_features=True returns a dict instead:
            "swin_tokens"         [B, 49, 768]
            "efficientnet_tokens" [B, 49, 1280]
            "fused_tokens"        [B, 49, 256]
            "fused"               [B, 256]
            "logits"              [B, 2]
            "attention"           [B, heads, 49, 49]  (only if return_attention=True)
        """
        swin_tokens = self.swin.forward_tokens(images)
        efficientnet_tokens = self.efficientnet.forward_tokens(images)

        if return_attention:
            fused_tokens, attention = self.fusion(
                swin_tokens, efficientnet_tokens, return_attention=True
            )
        else:
            fused_tokens = self.fusion(swin_tokens, efficientnet_tokens)
            attention = None

        # Mean pooling, identical to CrossAttentionFusion.forward_fused();
        # done here so the attention is not computed twice
        fused = fused_tokens.mean(dim=1)
        logits = self.classifier(fused)

        if not return_features:
            return logits

        outputs = {
            "swin_tokens": swin_tokens,
            "efficientnet_tokens": efficientnet_tokens,
            "fused_tokens": fused_tokens,
            "fused": fused,
            "logits": logits,
        }
        if attention is not None:
            outputs["attention"] = attention
        return outputs
