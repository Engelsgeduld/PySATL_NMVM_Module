from abc import abstractmethod
from typing import Self

from numpy import _typing

from src.algorithms import ALGORITHM_REGISTRY
from src.estimators.estimate_result import EstimateResult


class AbstractEstimator:
    def __init__(self, algorithm_name: str, params: dict | None = None) -> None:
        self.algorithm_name = algorithm_name
        if params is None:
            self.params = dict()
        else:
            self.params = params
        self.estimate_result = EstimateResult()
        self._registry = ALGORITHM_REGISTRY
        self._purpose = ""

    def get_params(self) -> dict:
        return {"algorithm_name": self.algorithm_name, "params": self.params, "estimated_result": self.estimate_result}

    def set_params(self, algorithm_name: str, params: dict | None = None) -> Self:
        self.algorithm_name = algorithm_name
        if params is None:
            self.params = dict()
        else:
            self.params = params
        return self

    def get_available_algorithms(self) -> list[tuple[str, str]]:
        return [key for key in self._registry.register_of_names.keys() if key[1] == self._purpose]

    def estimate(self, sample: _typing.ArrayLike) -> EstimateResult:
        cls = None
        if (self.algorithm_name, self._purpose) in self._registry.register_of_names:
            cls = self._registry.dispatch(self.algorithm_name, self._purpose)(sample, **self.params)
        if cls is None:
            raise ValueError("This algorithm does not exist")
        self.estimate_result = cls.algorithm(sample)
        return self.estimate_result
