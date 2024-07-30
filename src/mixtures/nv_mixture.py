from dataclasses import dataclass
from typing import Any

from scipy.stats import rv_continuous
from scipy.stats.distributions import rv_frozen

from src.mixtures.abstract_mixture import AbstractMixtures


@dataclass
class _NVMClassicDataCollector:
    """TODO: Change typing from float | int | etc to Protocol with __addition__ __multiplication__ __subtraction__"""

    """Data Collector for parameters of classical NVM"""
    alpha: float | int
    gamma: float | int
    distribution: rv_frozen | rv_continuous


@dataclass
class _NVMCanonicalDataCollector:
    """TODO: Change typing from float | int | etc to Protocol with __addition__ __multiplication__ __subtraction__"""

    """Data Collector for parameters of canonical NVM"""
    alpha: float | int
    distribution: rv_frozen | rv_continuous


class NormalVarianceMixtures(AbstractMixtures):

    _classical_collector = _NVMClassicDataCollector
    _canonical_collector = _NVMCanonicalDataCollector

    def __init__(self, mixture_form: str, **kwargs: Any) -> None:
        super().__init__(mixture_form, **kwargs)

    def compute_moment(self) -> Any:
        raise NotImplementedError("Must implement compute_moment")

    def compute_cdf(self) -> Any:
        raise NotImplementedError("Must implement cdf")

    def compute_pdf(self) -> Any:
        raise NotImplementedError("Must implement pdf")

    def compute_logpdf(self) -> Any:
        raise NotImplementedError("Must implement logpdf")
