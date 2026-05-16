import torch
import torch.nn as nn
from torchvision import models

IMG_SIZE = 224
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

CROSS_OUT_TYPES = ["SINGLE_LINE", "DOUBLE_LINE", "DIAGONAL", "CROSS",
                   "WAVE", "ZIG_ZAG", "SCRATCH"]
CROSS_OUT_LABELS = ["Single-Line", "Double-Line", "Diagonal", "Cross",
                    "Wave", "Zig-zag", "Scratch"]
NUM_CLASSES_TASK2 = len(CROSS_OUT_TYPES)

ATTENTION_LAYERS = 2
ATTENTION_HEADS = 8

# Model 1: EfficientNet-B0 single-task classifier
class Model1Net(nn.Module):
    """Single-task EfficientNet-B0. num_classes=1 for binary, 7 for multiclass."""

    def __init__(self, num_classes: int = 1):
        super().__init__()
        self.backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


# Model 2: EfficientNet-B0 + MTL heads
class Model2Net(nn.Module):
    """Shared EfficientNet backbone + parallel detection (1) and classification (7) heads."""

    def __init__(self, num_classes_task2: int = NUM_CLASSES_TASK2):
        super().__init__()
        self.backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.detection_head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, 1),
        )
        self.classification_head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes_task2),
        )

    def forward(self, x):
        feat = self.backbone(x)
        return self.detection_head(feat), self.classification_head(feat)


# Model 3: Model 2 + self-attention on the 7x7 feature map
class AttentionBlock(nn.Module):
    def __init__(self, feature_dim: int = 1280, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.position_embedding = nn.Parameter(torch.zeros(1, 49, feature_dim))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        self.norm_before_attention = nn.LayerNorm(feature_dim)
        self.norm_before_mlp = nn.LayerNorm(feature_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True)
        mlp_hidden = feature_dim * 2
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, feature_dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens, add_position: bool = True):
        if add_position:
            tokens = tokens + self.position_embedding
        attn_out, _ = self.attention(*[self.norm_before_attention(tokens)] * 3)
        tokens = tokens + attn_out
        tokens = tokens + self.mlp(self.norm_before_mlp(tokens))
        return tokens


class Model3Net(nn.Module):
    """Model 2 architecture + a stack of self-attention blocks on the 7x7 feature map."""

    def __init__(self, num_classes_task2: int = NUM_CLASSES_TASK2,
                 attention_layers: int = ATTENTION_LAYERS,
                 attention_heads: int = ATTENTION_HEADS):
        super().__init__()
        self.backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        feature_dim = self.backbone.classifier[1].in_features
        self.backbone.avgpool = nn.Identity()
        self.backbone.classifier = nn.Identity()
        self.attention_blocks = nn.ModuleList([
            AttentionBlock(feature_dim=feature_dim, num_heads=attention_heads)
            for _ in range(attention_layers)
        ])
        self.detection_head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(feature_dim, 1),
        )
        self.classification_head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(feature_dim, num_classes_task2),
        )

    def forward(self, x):
        feat_map = self.backbone.features(x)            # (B, 1280, 7, 7)
        tokens = feat_map.flatten(2).transpose(1, 2)    # (B, 49, 1280)
        for i, block in enumerate(self.attention_blocks):
            tokens = block(tokens, add_position=(i == 0))
        shared = tokens.mean(dim=1)
        return self.detection_head(shared), self.classification_head(shared)



# Factory + checkpoint loader
def build_model(model_idx: str, task: str = "multiclass") -> nn.Module:
    """Build an empty (no checkpoint) model.

    model_idx: "1", "2", or "3"
    task: only used for Model 1 ("binary" -> 1 output, "multiclass" -> 7)
    """
    if model_idx == "1":
        n = 1 if task == "binary" else NUM_CLASSES_TASK2
        return Model1Net(num_classes=n)
    if model_idx == "2":
        return Model2Net()
    if model_idx == "3":
        return Model3Net()
    raise ValueError(f"Unknown model_idx: {model_idx}")


def extract_state_dict(checkpoint: dict) -> dict:
    """Pull the model weights out of either checkpoint format used in the notebooks."""
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    # Already a raw state dict
    return checkpoint


def detect_task_from_checkpoint(checkpoint: dict) -> str:
    """Read the `task` metadata field saved by Model 1's notebook."""
    return checkpoint.get("task", "multiclass")
