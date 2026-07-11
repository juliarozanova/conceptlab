"""Interpretability methods under test, behind a common interface."""

from .base import InterpMethod, MethodContext, build_context
from .baselines import LinearProbeSkyline, PCAMethod, ICAMethod, LabelProbeImportance
from .sae import ReluSAE, TopKSAE
from .attribution import IntegratedGradients
from .causal import DirectionAblation, ActivationPatching

REGISTRY = {
    "probe_skyline": LinearProbeSkyline,
    "pca": PCAMethod,
    "ica": ICAMethod,
    "label_probe": LabelProbeImportance,
    "relu_sae": ReluSAE,
    "topk_sae": TopKSAE,
    "integrated_gradients": IntegratedGradients,
    "ablation": DirectionAblation,
    "patching": ActivationPatching,
}


def build_method(name: str, **kwargs) -> InterpMethod:
    if name not in REGISTRY:
        raise KeyError(f"unknown method '{name}'; known: {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)


__all__ = ["InterpMethod", "MethodContext", "build_context", "build_method", "REGISTRY"]
