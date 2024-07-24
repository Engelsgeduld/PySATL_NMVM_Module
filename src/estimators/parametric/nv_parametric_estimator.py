from numpy import _typing

from src.estimators.estimate_result import EstimateResult
from src.estimators.parametric.abstract_parametric_estimator import AbstractParametricEstimator


class NVParametricEstimator(AbstractParametricEstimator):
    def __init__(self, algorithm_name: str, params: dict | None = None) -> None:
        super().__init__(algorithm_name, params)
        ...

    def estimate(self, sample: _typing.ArrayLike) -> EstimateResult:
        return super().estimate(sample)
