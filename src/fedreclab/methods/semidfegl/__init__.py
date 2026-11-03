"""SemiDFEGL: semi-decentralized federated ego-graph learning."""

from src.fedreclab.methods.semidfegl.semidfegl import SemiDFEGL, default_param_ios
from src.fedreclab.methods.semidfegl.semidfegl_client import SemiDFEGLClientAlgorithm

__all__ = [
    "SemiDFEGL",
    "SemiDFEGLClientAlgorithm",
    "default_param_ios",
]
