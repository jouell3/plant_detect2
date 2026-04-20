# backend/app/src/herbs_detection/model_registry.py
"""
Single source of truth for all deployed model configurations.
Add or comment out entries here to enable/disable models globally.
"""
from dataclasses import dataclass

@dataclass(frozen=True)
class ModelConfig:
    key: str            # identifier used in API requests and wandb artifact names
    timm_name: str      # timm.create_model() identifier
    img_size: int       # native input resolution
    wandb_artifact: str # artifact name in the wandb registry (without :tag)
    enabled: bool = True

MODEL_REGISTRY: list[ModelConfig] = [
    ModelConfig(
        key="convnext_tiny",
        timm_name="convnext_tiny",
        img_size=224,
        wandb_artifact="convnext_tiny_best",
    ),
    ModelConfig(
        key="efficientnet_b3",
        timm_name="efficientnet_b3",
        img_size=300,
        wandb_artifact="efficientnet_b3_best",
    ),
    ModelConfig(
        key="efficientnet_b4",
        timm_name="efficientnet_b4",
        img_size=380,
        wandb_artifact="efficientnet_b4_best",
    ),
    ModelConfig(
        key="mobilenetv3_large",
        timm_name="mobilenetv3_large_100",
        img_size=224,
        wandb_artifact="mobilenetv3_large_best",
    ),
    ModelConfig(
        key="resnet50",
        timm_name="resnet50",
        img_size=224,
        wandb_artifact="resnet50_best",
    ),
]
    
# Fast lookup by key

REGISTRY_BY_KEY: dict[str, ModelConfig] = {m.key: m for m in MODEL_REGISTRY}
ENABLED_KEYS: list[str] = [m.key for m in MODEL_REGISTRY if m.enabled]
