from dataclasses import dataclass
from typing import Any

from scipy.stats import rv_continuous
from scipy.stats.distributions import rv_frozen

from src.mixtures.abstract_mixture import AbstractMixtures


@dataclass
class _NMVMClassicDataCollector:
    """TODO: Change typing from float | int | etc to Protocol with __addition__ __multiplication__ __subtraction__"""

    """Data Collector for parameters of classical NMVM"""
    alpha: float | int
    beta: float | int
    gamma: float | int
    distribution: rv_frozen | rv_continuous


@dataclass
class _NMVMCanonicalDataCollector:
    """TODO: Change typing from float | int | etc to Protocol with __addition__ __multiplication__ __subtraction__"""

    """Data Collector for parameters of canonical NMVM"""
    alpha: float | int
    mu: float | int
    distribution: rv_frozen | rv_continuous


class NormalMeanVarianceMixtures(AbstractMixtures):
    _classical_collector = _NMVMClassicDataCollector
    _canonical_collector = _NMVMCanonicalDataCollector

    def __init__(self, mixture_form: str, **kwargs: Any) -> None:
        super().__init__(mixture_form, **kwargs)

    def compute_moment(self) -> Any:
        pass

    def compute_cdf(self) -> Any:
        pass

    def compute_pdf(self) -> Any:
        pass

    def compute_logpdf(self) -> Any:
        pass
