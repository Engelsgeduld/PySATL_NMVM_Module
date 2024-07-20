from abc import ABCMeta, abstractmethod
from typing import Any

import scipy
from numpy import _typing

from src.register.register import Registry


class AbstractMixtures(metaclass=ABCMeta):
    """Base class for Mixtures"""

    def __init__(self) -> None:
        self.param_collector: Registry = Registry()
        self.semi_param_collector: Registry = Registry()

    @abstractmethod
    def classic_generate(
        self, size: int, w_distribution: scipy.stats.rv_continuous, params: list[float]
    ) -> _typing.ArrayLike:
        """Generate a samples of given size. Classical form of Mixture

        Args:
            size: length of sample
            w_distribution: Distribution of random value w
            params: Parameters of Mixture. For example: alpha, betta, gamma for NMM

        Returns: samples of given size

        """

    @abstractmethod
    def canonical_generate(
        self, size: int, w_distribution: scipy.stats.rv_continuous, params: list[float]
    ) -> _typing.ArrayLike:
        """Generate a samples of given size. Canonical form of Mixture

        Args:
            size: length of sample
            w_distribution: Distribution of random value w
            params: Parameters of Mixture. For example: theta for NMM

        Returns: samples of given size

        """

    @abstractmethod
    def param_algorithm(self, name: str, sample: _typing.ArrayLike, params: list[float]) -> Any:
        """Select and run parametric algorithm

        Args:
            name: Name of Algorithm
            sample: Vector of random values
            params: Parameters of Algorithm

        Returns: TODO

        """

    @abstractmethod
    def semi_param_algorithm(self, name: str, sample: _typing.ArrayLike, params: list[float]) -> Any:
        """Select and run semi-parametric algorithm

        Args:
            name: Name of Algorithm
            sample: Vector of random values
            params: Parameters of Algorithm

        Returns: TODO

        """
