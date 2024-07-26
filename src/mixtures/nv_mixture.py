from typing import Any

from scipy.stats import rv_continuous
from scipy.stats.distributions import rv_frozen

from src.mixtures.abstract_mixture import AbstractMixtures


class NormalVarianceMixtures(AbstractMixtures):

    def __init__(self, mixture_form: str, **kwargs: Any) -> None:
        if mixture_form == "classical":
            self._classical_params_validation(kwargs)
            self.alpha = kwargs["alpha"]
            self.gamma = kwargs["gamma"]
        elif mixture_form == "canonical":
            self._canonical_params_validation(kwargs)
            self.alpha = kwargs["alpha"]
        else:
            raise AssertionError(f"Unknown mixture form: {mixture_form}")
        self.distribution = kwargs["distribution"]

    @staticmethod
    def _classical_params_validation(params: dict) -> None:
        """Validation parameters for classic generate for NVM

        Args:
            params: Parameters of Mixture. For example: alpha, gamma, distribution for NVM

        Returns: None

        Raises:
            ValueError: If alpha not in kwargs
            ValueError: If gamma not in kwargs
            ValueError: If distribution not in kwargs
            ValueError: If distribution type is not rv_continuous

        """

        if len(params) != 3:
            raise ValueError("Expected 3 parameters")
        if "alpha" not in params:
            raise ValueError("Expected alpha")
        if "gamma" not in params:
            raise ValueError("expected gamma")
        if "distribution" not in params:
            raise ValueError("expected distribution")
        if not isinstance(params["distribution"], (rv_continuous, rv_frozen)):
            raise ValueError("Expected rv continuous distribution")

    @staticmethod
    def _canonical_params_validation(params: dict) -> None:
        """Validation parameters for canonical generate for NVM

        Args:
            params: Parameters of Mixture. For example: alpha, distribution for NVM

        Returns: None

        Raises:
            ValueError: If alpha not in kwargs
            ValueError: If distribution not in kwargs
            ValueError: If distribution type is not rv_continuous

        """

        if len(params) != 2:
            raise ValueError("Expected 2 parameter")
        if "alpha" not in params:
            raise ValueError("Expected alpha")
        if "distribution" not in params:
            raise ValueError("expected distribution")
        if not isinstance(params["distribution"], (rv_continuous, rv_frozen)):
            raise ValueError("Expected rv continuous distribution")
