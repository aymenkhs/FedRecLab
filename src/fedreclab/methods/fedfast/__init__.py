"""FedFast (Muhammad et al., KDD 2020)."""

from src.fedreclab.methods.fedfast.fedfast import FedFast, default_param_ios
from src.fedreclab.methods.fedfast.fedfast_client import FedFastClientAlgorithm
from src.fedreclab.methods.fedfast.fedfast_model import GeneralMatrixFactorization

__all__ = [
    "FedFast",
    "FedFastClientAlgorithm",
    "GeneralMatrixFactorization",
    "default_param_ios",
]
