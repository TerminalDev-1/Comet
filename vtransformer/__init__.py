"""VTransformer: a faster-to-train Transformer variant (see plan.md)."""
from .baseline import BaselineConfig, BaselineGPT

MODELS = {
    "baseline": (BaselineConfig, BaselineGPT),
}
