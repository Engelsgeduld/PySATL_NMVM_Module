from abc import abstractmethod

import numpy._typing as tpg

from src.mixtures.abstract_mixture import AbstractMixtures


class AbstractGenerator:
    @staticmethod
    @abstractmethod
    def canonical_generate(mixture: AbstractMixtures, size: int) -> tpg.NDArray: ...

    """Generate a sample of given size. Classical form of Mixture

    Args:
        mixture: NMM | NVM | NMVM
        size: length of sample

    Returns: sample of given size

    """

    @staticmethod
    @abstractmethod
    def classical_generate(mixture: AbstractMixtures, size: int) -> tpg.NDArray: ...

    """Generate a sample of given size. Canonical form of Mixture

    Args:
        mixture: NMM | NVM | NMVM
        size: length of sample

    Returns: sample of given size

    """
