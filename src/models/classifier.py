"""
Classification head: fused feature vector -> real/fake logits.

Input:
    fused: [batch_size, 256]   (CrossAttentionFusion.forward_fused output)

Output:
    logits: [batch_size, 2]    raw scores, NOT probabilities
        index 0 = FAKE
        index 1 = REAL
    (matches ImageFolder's class_to_idx in src/data/dataset.py)

Use softmax(logits) for probabilities. Training will use
nn.CrossEntropyLoss, which expects raw logits and applies softmax itself.

Architecture (a project choice, not taken from the DeFaX paper):
    Linear(256 -> 128) -> GELU -> Dropout(0.3) -> Linear(128 -> 2)

- One small hidden layer: enough to combine the fused features without
  adding many parameters (~33K).
- GELU: the same activation family used inside Swin Transformer.
- Dropout 0.3: the head is trained on top of large pretrained backbones and
  can overfit easily; dropout is active only in training mode.
- No extra normalization: the fused input already comes from a LayerNorm
  inside CrossAttentionFusion.
"""

from torch import nn

CLASS_NAMES = {0: "FAKE", 1: "REAL"}


class Classifier(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, num_classes=2, dropout=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, fused):
        """[B, input_dim] -> [B, num_classes] logits."""
        if fused.ndim != 2 or fused.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input [batch_size, {self.input_dim}], got {list(fused.shape)}"
            )
        return self.layers(fused)
