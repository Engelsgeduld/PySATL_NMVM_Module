from typing import Any

import scipy
from numpy import _typing

from src.mixtures.abstract_mixture import AbstractMixtures


class NormalVarianceMixtures(AbstractMixtures):

    def __init__(self) -> None:
        super().__init__()
        ...

    @staticmethod
    def _classic_generate_params_validation(params: list[float]) -> tuple[float, float]:
        """Validation parameters for classic generate for NVM

        Args:
            params: Parameters of Mixture. For example: alpha, gamma for NVM

        Returns:
            params: alpha, gamma for NVM

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
            params: alpha for NVM

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

    def param_algorithm(self, name: str, sample: _typing.ArrayLike, params: dict) -> Any:
        """Select and run parametric algorithm for NVM

        Args:
            name: Name of Algorithm
            sample: Vector of random values
            params: Parameters of Algorithm

        Returns: TODO

        """
        cls = self.param_collector.dispatch(name)(sample, params)
        return cls.algorithm(sample)

    def semi_param_algorithm(self, name: str, sample: _typing.ArrayLike, params: dict) -> Any:
        """Select and run semi-parametric algorithm for NVM

        Args:
            name: Name of Algorithm
            sample: Vector of random values
            params: Parameters of Algorithm

        Returns: TODO

        """
        cls = self.semi_param_collector.dispatch(name)(sample, **params)
        return cls.algorithm(sample)
