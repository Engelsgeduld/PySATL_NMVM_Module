import numpy._typing as tpg
import scipy

from src.generators.abstract_generator import AbstractGenerator
from src.mixtures.abstract_mixture import AbstractMixtures
from src.mixtures.nm_mixture import NormalMeanMixtures


class NMGenerator(AbstractGenerator):

    @staticmethod
    def classical_generate(mixture: AbstractMixtures, size: int) -> tpg.NDArray:
        """Generate a sample of given size. Classical form of NMM

        Args:
            size: length of sample
            w_distribution: Distribution of random value w
            params: Parameters of Mixture. For example: alpha, beta, gamma for NMM

        Returns: sample of given size

        """
        if not isinstance(mixture, NormalMeanMixtures):
            raise ValueError("Mixture must be NormalMeanMixtures")
        mixing_values = mixture.distribution.rvs(size=size)
        normal_values = scipy.stats.norm.rvs(size=size)
        return mixture.alpha + mixture.beta * mixing_values + mixture.gamma * normal_values

    @staticmethod
    def canonical_generate(mixture: AbstractMixtures, size: int) -> tpg.NDArray:
        """Generate a sample of given size. Canonical form of NMM

        Args:
            size: length of sample
            w_distribution: Distribution of random value w
            params: Parameters of Mixture. For example: sigma for NMM

        Returns: sample of given size

        """

        if not isinstance(mixture, NormalMeanMixtures):
            raise ValueError("Mixture must be NormalMeanMixtures")
        mixing_values = mixture.distribution.rvs(size=size)
        normal_values = scipy.stats.norm.rvs(size=size)
        return mixing_values + mixture.sigma * normal_values
