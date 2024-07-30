from numpy import _typing

from src.estimators.estimate_result import EstimateResult
from src.estimators.parametric.abstract_parametric_estimator import AbstractParametricEstimator
from src.register.algorithm_purpose import AlgorithmPurpose


class NMParametricEstimator(AbstractParametricEstimator):
    def __init__(self, algorithm_name: str, params: dict | None = None) -> None:
        super().__init__(algorithm_name, params)
        self._purpose = AlgorithmPurpose.NM_PARAMETRIC

    def estimate(self, sample: _typing.ArrayLike) -> EstimateResult:
        return super().estimate(sample)
