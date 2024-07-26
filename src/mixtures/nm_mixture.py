from typing import Any

from scipy.stats import rv_continuous
from scipy.stats.distributions import rv_frozen

from src.mixtures.abstract_mixture import AbstractMixtures


class NormalMeanMixtures(AbstractMixtures):
    def __init__(self, mixture_form: str, **kwargs: Any) -> None:
        if mixture_form == "classical":
            self._classical_params_validation(kwargs)
            self.alpha = kwargs["alpha"]
            self.beta = kwargs["beta"]
            self.gamma = kwargs["gamma"]
        elif mixture_form == "canonical":
            self._canonical_params_validation(kwargs)
            self.sigma = kwargs["sigma"]
        else:
            raise AssertionError(f"Unknown mixture form: {mixture_form}")
        self.distribution = kwargs["distribution"]

    @staticmethod
    def _classical_params_validation(params: dict[str, float | rv_continuous]) -> None:
        """Validation parameters for classic generate for NMM

        Args:
            params: Parameters of Mixture. For example: alpha, beta, gamma, distribution for NMM

        Returns: None

        Raises:
            ValueError: If alpha not in kwargs
            ValueError: If beta not in kwargs
            ValueError: If gamma not in kwargs
            ValueError: If distribution not in kwargs
            ValueError: If distribution type is not rv_continuous

        """

        if len(params) != 4:
            raise ValueError("Expected 4 parameters")
        if "alpha" not in params:
            raise ValueError("Expected alpha")
        if "beta" not in params:
            raise ValueError("expected beta")
        if "gamma" not in params:
            raise ValueError("expected gamma")
        if "distribution" not in params:
            raise ValueError("expected distribution")
        if not isinstance(params["distribution"], (rv_continuous, rv_frozen)):
            raise ValueError("Expected rv continuous distribution")

    @staticmethod
    def _canonical_params_validation(params: dict[str, float | rv_continuous]) -> None:
        """Validation parameters for canonical generate for NMM

        Args:
            params: Parameters of Mixture. For example: sigma, distribution for NMM

        Returns: None

        Raises:
            ValueError: If sigma not in kwargs
            ValueError: If distribution not in kwargs
            ValueError: If distribution type is not rv_continuous

        """

        if len(params) != 2:
            raise ValueError("Expected 2 parameter")
        if "sigma" not in params:
            raise ValueError("Expected sigma")
        if params["sigma"] < 0:
            raise ValueError("Expected parameter greater than or equal to zero")
        if "distribution" not in params:
            raise ValueError("expected distribution")
        if not isinstance(params["distribution"], (rv_continuous, rv_frozen)):
            raise ValueError("Expected rv continuous distribution")

    def compute_moment(self) -> Any:
        raise NotImplementedError("Must implement compute_moment")

    def compute_cdf(self) -> Any:
        raise NotImplementedError("Must implement cdf")

    def compute_pdf(self) -> Any:
        raise NotImplementedError("Must implement pdf")

    def compute_logpdf(self) -> Any:
        raise NotImplementedError("Must implement logpdf")
