from abc import abstractmethod

import numpy._typing as tpg

from src.mixtures.abstract_mixture import AbstractMixtures


class AbstractGenerator:
    @staticmethod
    @abstractmethod
    def canonical_generate(mixture: AbstractMixtures, size: int) -> tpg.NDArray: ...

    @staticmethod
    @abstractmethod
    def classical_generate(mixture: AbstractMixtures, size: int) -> tpg.NDArray: ...
