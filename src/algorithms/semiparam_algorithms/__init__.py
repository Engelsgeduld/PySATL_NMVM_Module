from src.algorithms.semiparam_algorithms.nvm_semi_param_algorithms.mu_estimation import SemiParametricMuEstimation
from src.register.register import Registry

SEMI_PARAM_ALGORITHM_REGISTRY: Registry = Registry()
SEMI_PARAM_ALGORITHM_REGISTRY.register("mu_estimation")(SemiParametricMuEstimation)
