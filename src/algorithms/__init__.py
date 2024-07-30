from src.algorithms.semiparam_algorithms.nvm_semi_param_algorithms.mu_estimation import SemiParametricMuEstimation
from src.register.algorithm_purpose import AlgorithmPurpose
from src.register.register import Registry

ALGORITHM_REGISTRY: Registry = Registry()
ALGORITHM_REGISTRY.register("mu_estimation", AlgorithmPurpose.NMV_SEMIPARAMETRIC)(SemiParametricMuEstimation)
