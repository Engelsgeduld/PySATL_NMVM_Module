import math
from typing import Any

import mpmath
import numpy as np
from numpy import _typing


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

    def __init__(self, sample: np._typing.NDArray = None, params: list[Any] = None):
        if sample is None:
            self.sample = np.array([])
        else:
            self.sample = sample
        self.m, self.tolerance, self.omega = self._set_default_params(params)

    def _set_default_params(self, params: list[Any] | None) -> tuple:
        """Setting parameters of algorithm

        Args:
            params: list of parameters

        Returns: tuple of parameters

        """
        default_m = 1000
        default_tolerance = 1 / 10**5
        default_omega = self._default_omega

        if params is None:
            return default_m, default_tolerance, default_omega
        elif len(params) == 1:
            if isinstance(params[0], int) and params[0] > 0:
                return params[0], default_tolerance, default_omega
            else:
                raise ValueError("Expected positive integer as parameter m")
        elif len(params) == 2:
            if isinstance(params[0], int) and params[0] > 0:
                if isinstance(params[1], (int, float)) and params[1] > 0:
                    return params[0], params[1], default_omega
                else:
                    raise ValueError("Expected positive float as parameter tolerance")
            else:
                raise ValueError("Expected positive integer as parameter m")
        elif len(params) == 3:
            if isinstance(params[0], int) and params[0] > 0:
                if isinstance(params[1], (int, float)) and params[1] > 0:
                    if callable(params[2]):
                        return params[0], params[1], params[2]
                    else:
                        raise ValueError("Expected callable object as parameter omega")
                else:
                    raise ValueError("Expected positive float as parameter tolerance")
            else:
                raise ValueError("Expected positive integer as parameter m")
        else:
            raise ValueError("Expected 1, 2, or 3 parameters")

    @staticmethod
    def _default_omega(x: float) -> float:
        """Default omega function

        Args:
            x: float

        Returns: function value

        """
        if abs(x) <= math.pi:
            return -1 * math.sin(x)
        return 0

    def __w(self, p: float, sample: np._typing.NDArray) -> float:
        """Root of this function is an estimation of mu

        Args:
            p: float
            sample: sample of the analysed distribution

        Returns: function value

        """
        y = 0
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

        left, right = 0, self.m
        it = 0
        while left <= right:
            mid = (right + left) / 2
            if it > 10**6:
                return mid
            it += 1
            if abs(self.__w(mid, sample)) < self.tolerance:
                return mid
            elif self.__w(mid, sample) < 0:
                left = mid
            else:
                right = mid
        return -1
