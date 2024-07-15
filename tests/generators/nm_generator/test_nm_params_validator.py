import sys

import numpy as np
import pytest

from src.mixtures.nm_mixture import *


class TestGenerateParamsValidators:
    @pytest.mark.parametrize(
        "params",
        [[], [1], [1.111, 2.111], [1, 1, 1, 1], np.random.uniform(-100, 100, size=(100, 1))],
    )
    def test_classic_generate_validator_value_error_length(self, params: list[float]) -> None:
        with pytest.raises(ValueError, match="Expected 3 parameters"):
            NormalMeanMixtures._classic_generate_params_validation(params)

    @pytest.mark.parametrize(
        "params",
        [
            [sys.float_info.max, sys.float_info.max, sys.float_info.max],
            [1, 1, 1],
            [0, 0, 0],
            [-1, -1, -1],
            [0.333, 0.333, 0.666],
        ],
    )
    def test_classic_generate_validator_correct(self, params: list[float]) -> None:
        NormalMeanMixtures._classic_generate_params_validation(params)

    @pytest.mark.parametrize("params", np.random.uniform(-100, 100, size=(50, 3)))
    def test_classic_generate_validator_correct_random(self, params: list[float]) -> None:
        NormalMeanMixtures._classic_generate_params_validation(params)

    @pytest.mark.parametrize(
        "params",
        [
            [],
            [1.111, 2.111],
            [1, 1, 1],
            [1, 1, 1, 1],
            np.random.uniform(1, 100, size=(100, 1)),
        ],
    )
    def test_canonical_generate_validator_value_error_length(self, params: list[float]) -> None:
        with pytest.raises(ValueError, match="Expected 1 parameter"):
            NormalMeanMixtures._canonical_generate_params_validation(params)

    @pytest.mark.parametrize(
        "params",
        [[-1], [-1000], [-9999]],
    )
    def test_canonical_generate_validator_value_error_sign(self, params: list[float]) -> None:
        with pytest.raises(ValueError, match="Expected parameter greater than or equal to zero"):
            NormalMeanMixtures._canonical_generate_params_validation(params)

    @pytest.mark.parametrize("params", [[sys.float_info.max], [1], [0], [0.333], [10000]])
    def test_canonical_generate_validator_correct(self, params: list[float]) -> None:
        NormalMeanMixtures._canonical_generate_params_validation(params)

    @pytest.mark.parametrize("params", np.random.uniform(1, 100, size=(50, 1)))
    def test_canonical_generate_validator_correct_random(self, params: list[float]) -> None:
        NormalMeanMixtures._canonical_generate_params_validation(params)
