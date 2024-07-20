from typing import Any

import scipy
from numpy import _typing

from src.mixtures.abstract_mixture import AbstractMixtures


class NormalMeanMixtures(AbstractMixtures):
    def __init__(self) -> None:
        super().__init__()
        ...

    @staticmethod
    def _classic_generate_params_validation(params: list[float]) -> tuple[float, float, float]:
        """Validation parameters for classic generate for NMM

        Args:
            params: Parameters of Mixture. For example: alpha, beta, gamma for NMM

        Returns:
            params: alpha, beta, gamma for NMM

        """
        if len(params) != 3:
            raise ValueError("Expected 3 parameters")
        alpha, beta, gamma = params
        return alpha, beta, gamma

    @staticmethod
    def _canonical_generate_params_validation(params: list[float]) -> float:
        """Validation parameters for canonical generate for NMM

        Args:
            params: Parameters of Mixture. For example: sigma for NMM

        Returns:
            params: sigma for NMM

        """
        if len(params) != 1:
            raise ValueError("Expected 1 parameter")
        sigma = params[0]
        if sigma < 0:
            raise ValueError("Expected parameter greater than or equal to zero")
        return sigma

    def classic_generate(
        self, size: int, w_distribution: scipy.stats.rv_continuous, params: list[float]
    ) -> _typing.ArrayLike:
        """Generate a sample of given size. Classical form of NMM

        Args:
            size: length of sample
            w_distribution: Distribution of random value w
            params: Parameters of Mixture. For example: alpha, beta, gamma for NMM

        Returns: sample of given size

        """
        alpha, beta, gamma = self._classic_generate_params_validation(params)
        mixing_values = w_distribution.rvs(size=size)
        normal_values = scipy.stats.norm.rvs(size=size)
        return alpha + beta * mixing_values + gamma * normal_values

    def canonical_generate(
        self, size: int, w_distribution: scipy.stats.rv_continuous, params: list[float]
    ) -> _typing.ArrayLike:
        """Generate a sample of given size. Canonical form of NMM

        Args:
            size: length of sample
            w_distribution: Distribution of random value w
            params: Parameters of Mixture. For example: sigma for NMM

        Returns: sample of given size

        """
        sigma = self._canonical_generate_params_validation(params)
        mixing_values = w_distribution.rvs(size=size)
        normal_values = scipy.stats.norm.rvs(size=size)
        return mixing_values + sigma * normal_values

    def param_algorithm(self, name: str, sample: _typing.ArrayLike, params: list[float]) -> Any:
        """Select and run parametric algorithm for NMM

        Args:
            name: Name of Algorithm
            sample: Vector of random values
            params: Parameters of Algorithm

        Returns: TODO

        """
        cls = self.param_collector.dispatch(name)(sample, params)
        return cls.algorithm(sample)

    def semi_param_algorithm(self, name: str, sample: _typing.ArrayLike, params: list[float]) -> Any:
        """Select and run semi-parametric algorithm for NMM

        Args:
            name: Name of Algorithm
            sample: Vector of random values
            params: Parameters of Algorithm

        Returns: TODO

        """
        cls = self.semi_param_collector.dispatch(name)(sample, params)
        return cls.algorithm(sample)
