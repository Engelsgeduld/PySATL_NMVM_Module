import math
from typing import Callable, TypedDict, Unpack

import mpmath
import numpy as np
from numpy import _typing

M_DEFAULT_VALUE = 1000
TOLERANCE_DEFAULT_VALUE = 10**-5
OMEGA_DEFAULT_VALUE = lambda x: -1 * math.sin(x) if abs(x) <= math.pi else 0
MAX_ITERATIONS_DEFAULT_VALUE = 10**9


class SemiParametricMuEstimation:
    """Estimation of mu parameter of NVM mixture represented in canonical form Y = alpha + mu*xi + sqrt(xi)*N,
    where alpha = 0

    Args:
        sample: sample of the analysed distribution
        params: parameters of the algorithm
                m - search area,
                tolerance - defines where to stop binary search,
                omega - Lipschitz continuous odd function on R with compact support

    """

    class ParamsAnnotation(TypedDict, total=False):
        """Class for parameters annotation"""
        m: float
        tolerance: float
        omega: Callable[[float], float]
        max_iterations: float

    def __init__(self, sample: _typing.ArrayLike = None, **kwargs: Unpack[ParamsAnnotation]):
        self.sample = np.array([]) if sample is None else sample
        self.m, self.tolerance, self.omega, self.max_iterations = self._validate_kwargs(**kwargs)

    def _validate_kwargs(
        self, **kwargs: Unpack[ParamsAnnotation]
    ) -> tuple[float, float, Callable[[float], float], float]:
        """Parameters validation function

        Args:
            kwargs: Parameters of Algorithm

        Returns: Parameters of Algorithm

        """
        if any([i not in self.ParamsAnnotation.__annotations__ for i in kwargs]):
            raise ValueError("Got unexpected parameter")
        if "m" in kwargs and (not isinstance(kwargs.get("m"), int) or kwargs.get("m", -1) <= 0):
            raise ValueError("Expected positive integer as parameter m")
        if "tolerance" in kwargs and (
            not isinstance(kwargs.get("tolerance"), (int, float)) or kwargs.get("tolerance", -1) <= 0
        ):
            raise ValueError("Expected positive float as parameter tolerance")
        if "omega" in kwargs and not callable(kwargs.get("omega")):
            raise ValueError("Expected callable object as parameter omega")
        if "max_iterations" in kwargs and (
            not isinstance(kwargs.get("max_iterations"), int) or kwargs.get("max_iterations", -1) <= 0
        ):
            raise ValueError("Expected positive integer as parameter max_iterations")
        return (
            kwargs.get("m", M_DEFAULT_VALUE),
            kwargs.get("tolerance", TOLERANCE_DEFAULT_VALUE),
            kwargs.get("omega", OMEGA_DEFAULT_VALUE),
            kwargs.get("max_iterations", MAX_ITERATIONS_DEFAULT_VALUE),
        )

    def __w(self, p: float, sample: np._typing.NDArray) -> float:
        """Root of this function is an estimation of mu

        Args:
            p: float
            sample: sample of the analysed distribution

        Returns: function value

        """
        y = 0.0
        for x in sample:
            try:
                e = math.exp(-p * x)
            except OverflowError:
                e = mpmath.exp(-p * x)
            y += e * self.omega(x)
        return y

    def algorithm(self, sample: np._typing.NDArray) -> float:
        """Root of this function is an estimation of mu

        Args:
            sample: sample of the analysed distribution

        Returns: estimated mu value

        """

        if self.__w(0, sample) == 0:
            return 0
        if self.__w(0, sample) > 0:
            return -1 * self.algorithm(-1 * sample)
        if self.__w(self.m, sample) < 0:
            return self.m

        left, right = 0.0, self.m
        iteration = 0
        while left <= right:
            mid = (right + left) / 2
            if iteration > self.max_iterations:
                return mid
            iteration += 1
            if abs(self.__w(mid, sample)) < self.tolerance:
                return mid
            elif self.__w(mid, sample) < 0:
                left = mid
            else:
                right = mid
        return -1
