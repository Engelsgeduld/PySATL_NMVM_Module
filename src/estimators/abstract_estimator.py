from abc import abstractmethod
from typing import Self

from numpy import _typing

from src.estimators.estimate_result import EstimateResult
from src.register.register import Registry


class AbstractEstimator:
    def __init__(self, algorithm_name: str, params: dict | None = None) -> None:
        self.algorithm_name = algorithm_name
        if params is None:
            self.params = dict()
        else:
            self.params = params
        self.estimate_result = EstimateResult()

    def get_params(self) -> dict:
        return {"algorithm_name": self.algorithm_name, "params": self.params}

    def set_params(self, algorithm_name: str, params: dict | None = None) -> Self:
        self.algorithm_name = algorithm_name
        if params is None:
            self.params = dict()
        else:
            self.params = params
        return self

    @abstractmethod
    def get_available_algorithms(self) -> list[str]: ...

    @abstractmethod
    def estimate(self, sample: _typing.ArrayLike) -> EstimateResult: ...
