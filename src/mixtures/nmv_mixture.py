from typing import Any

import scipy
from numpy import _typing

from src.mixtures.abstract_mixture import AbstractMixtures
from src.register.register import Registry


class NormalMeanVarianceMixtures(AbstractMixtures):

    def __init__(self, param_collector: Registry, semi_param_collector: Registry) -> None:
        """

        Args:
            param_collector: Collector of implementations of parametric algorithms
            semi_param_collector: Collector of implementations of semi-parametric algorithms

        """
        super().__init__(param_collector, semi_param_collector)
        ...

    def classic_generate(
        self, size: int, w_distribution: scipy.stats.rv_continuous, params: list[float]
    ) -> _typing.ArrayLike:
        """Generate a sample of given size. Classical form of NMVM

        Args:
            size: length of sample
            w_distribution: Distribution of random value w
            params: Parameters of Mixture. For example: alpha, beta, gamma for NMVM

        Returns: sample of given size

        """
        mixing_values = w_distribution.rvs(size=size)
        normal_values = scipy.stats.norm.rvs(size=size)
        return params[0] + params[1] * mixing_values + params[2] * (mixing_values**0.5) * normal_values

    def canonical_generate(
        self, size: int, w_distribution: scipy.stats.rv_continuous, params: list[float]
    ) -> _typing.ArrayLike:
        """Generate a sample of given size. Canonical form of NMVM

        Args:
            size: length of sample
            w_distribution: Distribution of random value w
            params: Parameters of Mixture. For example: alpha, mu for NMVM

        Returns: sample of given size

        """
        mixing_values = w_distribution.rvs(size=size)
        normal_values = scipy.stats.norm.rvs(size=size)
        return params[0] + params[1] * mixing_values + (mixing_values**0.5) * normal_values

    def param_algorithm(self, name: str, selection: _typing.ArrayLike, params: list[float]) -> Any:
        """Select and run parametric algorithm for NMVM

        Args:
            name: Name of Algorithm
            selection: Vector of random values
            params: Parameters of Algorithm

        Returns: TODO

        """
        ...

    def semi_param_algorithm(self, name: str, selection: _typing.ArrayLike, params: list[float]) -> Any:
        """Select and run semi-parametric algorithm for NMVM

        Args:
            name: Name of Algorithm
            selection: Vector of random values
            params: Parameters of Algorithm

        Returns: TODO

        """
        ...
