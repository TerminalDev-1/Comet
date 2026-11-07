"""Comet: a faster-to-train Transformer variant (see plan.md)."""
from .baseline import BaselineConfig, BaselineGPT
from .model import Comet, CometConfig

MODELS = {
    "baseline": (BaselineConfig, BaselineGPT),
    "comet": (CometConfig, Comet),
    "vtransformer": (CometConfig, Comet),  # old name, so earlier checkpoints still load
}
