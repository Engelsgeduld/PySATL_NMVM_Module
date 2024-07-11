from abc import ABCMeta, abstractmethod
from typing import Any

import scipy
from numpy import _typing

from src.register.register import Registry


class AbstractMixtures(metaclass=ABCMeta):
    def __init__(self, param_collector: Registry, semi_param_collector: Registry) -> None:
        self.param_collector = param_collector
        self.semi_param_collector = semi_param_collector

    @abstractmethod
    def generate(
        self, size: int, w_distribution: scipy.stats.rv_continuous, params: list[float]
    ) -> _typing.ArrayLike: ...

    @abstractmethod
    def param_algorithm(self, name: str, selection: _typing.ArrayLike, params: list[float]) -> Any: ...

    @abstractmethod
    def semi_param_algorithm(self, name: str, selection: _typing.ArrayLike, params: list[float]) -> Any: ...
