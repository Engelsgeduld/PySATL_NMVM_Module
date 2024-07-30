from numpy import _typing

from src.estimators.estimate_result import EstimateResult
from src.estimators.semiparametric.abstract_semiparametric_estimator import AbstractSemiParametricEstimator
from src.register.algorithm_purpose import AlgorithmPurpose


class NMVSemiParametricEstimator(AbstractSemiParametricEstimator):
    def __init__(self, algorithm_name: str, params: dict | None = None) -> None:
        super().__init__(algorithm_name, params)
        self._purpose = AlgorithmPurpose.NMV_SEMIPARAMETRIC

    def estimate(self, sample: _typing.ArrayLike) -> EstimateResult:
        return super().estimate(sample)
