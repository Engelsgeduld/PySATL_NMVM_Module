from abc import ABCMeta, abstractmethod
from typing import Any

from scipy.stats import rv_continuous


class AbstractMixtures(metaclass=ABCMeta):
    """Base class for Mixtures"""

    @abstractmethod
    def __init__(self, mixture_form: str, **kwargs: Any) -> None: ...

    @staticmethod
    @abstractmethod
    def _canonical_args_validation(params: dict) -> None: ...

    @staticmethod
    @abstractmethod
    def _classical_args_validation(params: dict) -> None: ...

    @abstractmethod
    def compute_moment(self) -> Any: ...

    @abstractmethod
    def compute_cdf(self) -> Any: ...

    @abstractmethod
    def compute_pdf(self) -> Any: ...

    @abstractmethod
    def compute_logpdf(self) -> Any: ...
