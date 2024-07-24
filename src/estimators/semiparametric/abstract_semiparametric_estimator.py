from numpy import _typing

from src.algorithms.semiparam_algorithms import SEMI_PARAM_ALGORITHM_REGISTRY
from src.estimators.abstract_estimator import AbstractEstimator
from src.estimators.estimate_result import EstimateResult

REGISTRY = SEMI_PARAM_ALGORITHM_REGISTRY


class AbstractSemiParametricEstimator(AbstractEstimator):
    def __init__(self, algorithm_name: str, params: dict | None = None) -> None:
        super().__init__(algorithm_name, params)
        self.registry = SEMI_PARAM_ALGORITHM_REGISTRY

    def get_available_algorithms(self) -> list[str]:
        return list(self.registry.register_of_names.keys())

    def estimate(self, sample: _typing.ArrayLike) -> EstimateResult:
        cls = None
        if self.algorithm_name in self.registry.register_of_names:
            cls = self.registry.dispatch(self.algorithm_name)(sample, **self.params)
        if cls is None:
            raise ValueError("This algorithm does not exist")
        self.estimate_result = cls.algorithm(sample)
        return self.estimate_result
