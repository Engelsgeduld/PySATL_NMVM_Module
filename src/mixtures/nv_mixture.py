from typing import Any

import scipy
from numpy import _typing

from src.mixtures.abstract_mixture import AbstractMixtures
from src.register.register import Registry


class NormalVarianceMixtures(AbstractMixtures):

    def __init__(self, param_collector: Registry, semi_param_collector: Registry) -> None:
        """

        Args:
            param_collector: Collector of implementations of parametric algorithms
            semi_param_collector: Collector of implementations of semi-parametric algorithms

        """
        super().__init__(param_collector, semi_param_collector)
        ...

    @staticmethod
    def _classic_generate_params_validation(params: list[float]) -> tuple[float, float]:
        """Validation parameters for classic generate for NVM

        Args:
            params: Parameters of Mixture. For example: alpha, gamma for NVM

        Returns:

        """
        if len(params) != 2:
            raise ValueError("Expected 2 parameters")
        alpha, gamma = params
        return alpha, gamma

    @staticmethod
    def _canonical_generate_params_validation(params: list[float]) -> float:
        """Validation parameters for canonical generate for NVM

        Args:
            params: Parameters of Mixture. For example: alpha for NVM

        Returns:

        """
        if len(params) != 1:
            raise ValueError("Expected 1 parameter")
        alpha = params[0]
        return alpha

    def classic_generate(
        self, size: int, w_distribution: scipy.stats.rv_continuous, params: list[float]
    ) -> _typing.ArrayLike:
        """Generate a sample of given size. Classical form of NVM

        Args:
            size: length of sample
            w_distribution: Distribution of random value w
            params: Parameters of Mixture. For example: alpha, gamma for NVM

        Returns: sample of given size

        """
        alpha, gamma = self._classic_generate_params_validation(params)
        mixing_values = w_distribution.rvs(size=size)
        normal_values = scipy.stats.norm.rvs(size=size)
        return alpha + gamma * (mixing_values**0.5) * normal_values

    def canonical_generate(
        self, size: int, w_distribution: scipy.stats.rv_continuous, params: list[float]
    ) -> _typing.ArrayLike:
        """Generate a sample of given size. Canonical form of NVM

        Args:
            size: length of sample
            w_distribution: Distribution of random value w
            params: Parameters of Mixture. For example: alpha for NVM

        Returns: sample of given size

        """
        alpha = self._canonical_generate_params_validation(params)
        mixing_values = w_distribution.rvs(size=size)
        normal_values = scipy.stats.norm.rvs(size=size)
        return alpha + (mixing_values**0.5) * normal_values

    def param_algorithm(self, name: str, selection: _typing.ArrayLike, params: list[float]) -> Any:
        """Select and run parametric algorithm for NVM

        Args:
            name: Name of Algorithm
            selection: Vector of random values
            params: Parameters of Algorithm

        Returns: TODO

        """
        ...

    def semi_param_algorithm(self, name: str, selection: _typing.ArrayLike, params: list[float]) -> Any:
        """Select and run semi-parametric algorithm for NVM

        Args:
            name: Name of Algorithm
            selection: Vector of random values
            params: Parameters of Algorithm

        Returns: TODO

        """
        ...
