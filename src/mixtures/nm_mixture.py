from dataclasses import dataclass
from typing import Any

from scipy.stats import rv_continuous
from scipy.stats.distributions import rv_frozen

from src.mixtures.abstract_mixture import AbstractMixtures


@dataclass
class _NMMClassicDataCollector:
    """TODO: Change typing from float | int | etc to Protocol with __addition__ __multiplication__ __subtraction__"""

    """Data Collector for parameters of classical NMM"""
    alpha: float | int
    beta: float | int
    gamma: float | int
    distribution: rv_frozen | rv_continuous


@dataclass
class _NMMCanonicalDataCollector:
    """TODO: Change typing from float | int | etc to Protocol with __addition__ __multiplication__ __subtraction__"""

    """Data Collector for parameters of canonical NMM"""
    sigma: float | int
    distribution: rv_frozen | rv_continuous


class NormalMeanMixtures(AbstractMixtures):
    _classical_collector = _NMMClassicDataCollector
    _canonical_collector = _NMMCanonicalDataCollector

    def __init__(self, mixture_form: str, **kwargs: Any) -> None:
        """
        Read Doc of Parent Method
        """

        super().__init__(mixture_form, **kwargs)

    def _params_validation(self, data_collector: Any, params: dict[str, float | rv_continuous | rv_frozen]) -> Any:
        """
        Read parent method doc

        Raises:
            ValueError: If canonical Mixture has negative sigma parameter

        """

        data_class = super()._params_validation(data_collector, params)
        if hasattr(data_class, "sigma") and data_class.sigma < 0:
            raise ValueError("Sigma is negative")
        return data_class

    def compute_moment(self) -> Any:
        raise NotImplementedError("Must implement compute_moment")

    def compute_cdf(self) -> Any:
        raise NotImplementedError("Must implement cdf")

    def compute_pdf(self) -> Any:
        raise NotImplementedError("Must implement pdf")

    def compute_logpdf(self) -> Any:
        raise NotImplementedError("Must implement logpdf")
