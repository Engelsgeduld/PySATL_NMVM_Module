import timeit
from typing import Any, Callable

import numpy as np
import numpy._typing as tpg
import scipy


class RQMC:
    """Randomize Quasi Monte Carlo Method

    Args:
        func: integrated function
        error_tolerance: pre-specified error tolerance
        count: number of rows of random values matrix
        base_n: number of columns of random values matrix
        i_max: allowed number of cycles
        a: parameter for quantile of normal distribution

    """

    def __init__(
        self,
        func: Callable,
        error_tolerance: float = 1e-6,
        count: int = 25,
        base_n: int = 2**4,
        i_max: int = 600,
        a: float = 0.00047,
    ):
        self.func = func
        self.error_tolerance = error_tolerance
        self.count = count
        self.base_n = base_n
        self.i_max = i_max
        self.z = scipy.stats.norm.ppf(1 - a / 2)

    def independent_estimator(self, values: np._typing.NDArray) -> float:
        """Apply function to row of matrix and find mean of row

        Args:
            values: row of random values matrix

        Returns: mean of row

        """
        vfunc = np.vectorize(self.func)
        return 1 / len(values) * np.sum(vfunc(values))

    def estimator(self, random_matrix: np._typing.NDArray) -> tuple[float, np._typing.NDArray]:
        """Find mean of all rows

        Args:
            random_matrix: matrix of random values

        Returns: mean of all rows

        """
        values = np.array(list(map(self.independent_estimator, random_matrix)))
        return 1 / self.count * np.sum(values), values

    def update_independent_estimator(self, i: int, old_value: float, new_values: np._typing.NDArray) -> float:
        """Update mean of row

        Args:
            i: step count
            old_value: previous value of row on i-1 step
            new_values: new generated row of random values

        Returns: Updated mean of row

        """
        return (i * old_value + (i + self.base_n) * self.independent_estimator(new_values[: i * self.base_n])) / (
            2 * i + self.base_n
        )

    def update(
        self, j: int, old_values: np._typing.NDArray, random_matrix: np._typing.NDArray
    ) -> tuple[float, np._typing.NDArray]:
        """Update mean of all rows

        Args:
            j: step count
            old_values: previous values of row on i-1 step
            random_matrix: new generated matrix of random values

        Returns:Updated mean of all rows

        """
        values = []
        sum_of_new: float = 0.0
        for i in range(self.count):
            old_value, new_values = old_values[i], random_matrix[i]
            value = self.update_independent_estimator(j, old_value, new_values)
            values.append(value)
            sum_of_new += value
        values = np.array(values)
        return (1 / self.count) * sum_of_new, values

    def sigma(self, values: np._typing.NDArray, approximation: float) -> float:
        """Calculate parameter sigma for estimation

        Args:
            values:
            approximation:

        Returns: return sigma parameter

        """
        diff = np.sum(np.power(values - approximation, 2))
        return np.sqrt(1 / (self.count - 1) * diff)

    def rqmc(self) -> float:
        """Main function of algorithm

        Returns: approximation for integral of function from 0 to 1

        """
        sample = np.random.rand(self.count, self.base_n)
        approximation, values = self.estimator(sample)
        current_error_tolerance = self.sigma(values, approximation) * self.z
        for i in range(1, self.i_max):
            if current_error_tolerance < self.error_tolerance:
                return approximation
            sample = np.random.rand(self.count, self.base_n * i)
            approximation, values = self.update(i * self.base_n, values, sample)
            current_error_tolerance = self.sigma(values, approximation) * self.z
        return approximation

    def __call__(self) -> float:
        """Interface for users

        Returns: approximation for integral of function from 0 to 1

        """
        return self.rqmc()
