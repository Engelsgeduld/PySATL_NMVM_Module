from numpy import _typing

from src.estimators.estimate_result import EstimateResult
from src.estimators.semiparametric.abstract_semiparametric_estimator import AbstractSemiParametricEstimator


class NVSemiParametricEstimator(AbstractSemiParametricEstimator):
    def __init__(self, algorithm_name: str, params: dict | None = None) -> None:
        super().__init__(algorithm_name, params)
        self._purpose = "NVSemiparametric"

    def estimate(self, sample: _typing.ArrayLike) -> EstimateResult:
        return super().estimate(sample)
