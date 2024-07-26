from typing import Any

from scipy.stats import rv_continuous
from scipy.stats.distributions import rv_frozen

from src.mixtures.abstract_mixture import AbstractMixtures


class NormalMeanVarianceMixtures(AbstractMixtures):
    def __init__(self, mixture_form: str, **kwargs: Any) -> None:
        if mixture_form == "classical":
            self._classical_params_validation(kwargs)
            self.alpha = kwargs["alpha"]
            self.beta = kwargs["beta"]
            self.gamma = kwargs["gamma"]
        elif mixture_form == "canonical":
            self._canonical_params_validation(kwargs)
            self.alpha = kwargs["alpha"]
            self.mu = kwargs["mu"]
        else:
            raise AssertionError(f"Unknown mixture form: {mixture_form}")
        self.distribution = kwargs["distribution"]

    def compute_moment(self) -> Any:
        pass

    def compute_cdf(self) -> Any:
        pass

    def compute_pdf(self) -> Any:
        pass

    def compute_logpdf(self) -> Any:
        pass

    @staticmethod
    def _classical_params_validation(params: dict) -> None:
        """Validation parameters for classic generate for NMVM

        Args:
            params: Parameters of Mixture. For example: alpha, beta, gamma, distribution for NMVM

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
    def _canonical_params_validation(params: dict) -> None:
        """Validation parameters for canonical generate for NMVM

        Args:
            params: Parameters of Mixture. For example: alpha, mu, distribution for NMVM

        Returns: None

        Raises:
            ValueError: If alpha not in kwargs
            ValueError: If mu not in kwargs
            ValueError: If distribution not in kwargs
            ValueError: If distribution type is not rv_continuous

        """

        if len(params) != 3:
            raise ValueError("Expected 3 parameters")
        if "alpha" not in params:
            raise ValueError("Expected alpha")
        if "mu" not in params:
            raise ValueError("expected mu")
        if "distribution" not in params:
            raise ValueError("expected distribution")
        if not isinstance(params["distribution"], (rv_continuous, rv_frozen)):
            raise ValueError("Expected rv continuous distribution")
