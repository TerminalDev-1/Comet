"""VTransformer: a faster-to-train Transformer variant (see plan.md)."""
from .baseline import BaselineConfig, BaselineGPT
from .model import VConfig, VTransformer

MODELS = {
    "baseline": (BaselineConfig, BaselineGPT),
    "vtransformer": (VConfig, VTransformer),
}
