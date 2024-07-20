import math

import numpy as np
import pytest

from src.algorithms.nvm_semi_param_algorithms.mu_estimation import SemiParametricMuEstimation


def _test_omega(x: float) -> float:
    return (-(x**3)) * math.exp(-(1 / (1 - x**2))) if abs(x) < 1 else 0


class TestSetDefaultParams:

    @pytest.mark.parametrize(
        "params",
        [[], [1, 2, 3, 4], [1, 1, 1, 1, 1]],
    )
    def test_set_default_params_value_error_length(self, params: list[float]) -> None:
        with pytest.raises(ValueError, match="Expected 1, 2, or 3 parameters"):
            SemiParametricMuEstimation()._set_default_params(params)

    @pytest.mark.parametrize(
        "params",
        [[-10], [-0.5], [0], [0.5], ["str"]],
    )
    def test_set_default_params_len_1_value_error_m(self, params: list[float]) -> None:
        with pytest.raises(ValueError, match="Expected positive integer as parameter m"):
            SemiParametricMuEstimation()._set_default_params(params)

    @pytest.mark.parametrize(
        "params",
        [[-10, 1 / 10], [-0.5, 1 / 10], [0, 1 / 10], [0.5, 1 / 10], [[], 1 / 10], ["str", 1 / 10]],
    )
    def test_set_default_params_len_2_value_error_m(self, params: list[float]) -> None:
        with pytest.raises(ValueError, match="Expected positive integer as parameter m"):
            SemiParametricMuEstimation()._set_default_params(params)

    @pytest.mark.parametrize(
        "params",
        [[100, -1], [100, -0.5], [100, 0], [100, []], [100, "str"]],
    )
    def test_set_default_params_len_2_value_error_tol(self, params: list[float]) -> None:
        with pytest.raises(ValueError, match="Expected positive float as parameter tolerance"):
            SemiParametricMuEstimation()._set_default_params(params)

    @pytest.mark.parametrize(
        "params",
        [
            [-10, 1 / 10, _test_omega],
            [-0.5, 1 / 10, _test_omega],
            [0, 1 / 10, _test_omega],
            [0.5, 1 / 10, _test_omega],
            [[], 1 / 10, _test_omega],
            ["str", 1 / 10, _test_omega],
        ],
    )
    def test_set_default_params_len_3_value_error_m(self, params: list[float]) -> None:
        with pytest.raises(ValueError, match="Expected positive integer as parameter m"):
            SemiParametricMuEstimation()._set_default_params(params)

    @pytest.mark.parametrize(
        "params",
        [
            [100, -1, _test_omega],
            [100, -0.5, _test_omega],
            [100, 0, _test_omega],
            [100, [], _test_omega],
            [100, "str", _test_omega],
        ],
    )
    def test_set_default_params_len_3_value_error_tol(self, params: list[float]) -> None:
        with pytest.raises(ValueError, match="Expected positive float as parameter tolerance"):
            SemiParametricMuEstimation()._set_default_params(params)

    @pytest.mark.parametrize("params", [[100, 1 / 10, 1], [100, 1 / 10, []], [100, 1 / 10, "str"]])
    def test_set_default_params_len_3_value_error_omega(self, params: list[float]) -> None:
        with pytest.raises(ValueError, match="Expected callable object as parameter omega"):
            SemiParametricMuEstimation()._set_default_params(params)

    @pytest.mark.parametrize("params", [[100, 1 / 10, _test_omega], [1000, 10**-9, _test_omega], [1, 1, _test_omega]])
    def test_set_default_params_len_3_correct(self, params: list[float]) -> None:
        SemiParametricMuEstimation()._set_default_params(params)

    @pytest.mark.parametrize("params", [[100, 1 / 10], [1000, 10**-9], [1, 1]])
    def test_set_default_params_len_2_correct(self, params: list[float]) -> None:
        SemiParametricMuEstimation()._set_default_params(params)

    @pytest.mark.parametrize("params", [[100], [1000], [1]])
    def test_set_default_params_len_1_correct(self, params: list[float]) -> None:
        SemiParametricMuEstimation()._set_default_params(params)

    @pytest.mark.parametrize("params", [[100, 1 / 10, _test_omega], [1000, 10**-9, _test_omega], [1, 1, _test_omega]])
    def test_init_set_default_params_len_3_correct(self, params: list[float]) -> None:
        SemiParametricMuEstimation(np.array([1]), params)

    @pytest.mark.parametrize("params", [[100, 1 / 10], [1000, 10**-9], [1, 1]])
    def test_init_set_default_params_len_2_correct(self, params: list[float]) -> None:
        SemiParametricMuEstimation(np.array([1]), params)

    @pytest.mark.parametrize("params", [[100], [1000], [1]])
    def test_init_set_default_params_len_1_correct(self, params: list[float]) -> None:
        SemiParametricMuEstimation(np.array([1]), params)
