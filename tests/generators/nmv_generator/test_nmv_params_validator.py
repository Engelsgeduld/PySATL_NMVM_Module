import sys

import numpy as np
import pytest

from src.mixtures.nmv_mixture import *


class TestGenerateParamsValidators:
    @pytest.mark.parametrize(
        "params",
        [
            [],
            [0],
            [1.111, 2.111],
            [1, 1, 1, 1],
            np.random.uniform(0, 100, size=(100, 1)),
            np.random.uniform(0, 100, size=(100, 100)),
        ],
    )
    def test_classic_generate_validator_value_error_length(self, params: list[float]) -> None:
        with pytest.raises(ValueError, match="Expected 3 parameters"):
            NormalMeanVarianceMixtures._classic_generate_params_validation(params)

    @pytest.mark.parametrize(
        "params",
        [
            [sys.float_info.max, sys.float_info.max, sys.float_info.max],
            [sys.float_info.min, sys.float_info.min, sys.float_info.min],
            [1, 1, 1],
            [0, 0, 0],
            [-1, -1, -1],
            [0.333, -0.333, 0.666],
        ],
    )
    def test_classic_generate_validator_correct(self, params: list[float]) -> None:
        NormalMeanVarianceMixtures._classic_generate_params_validation(params)

    @pytest.mark.parametrize("params", np.random.uniform(-100, 100, size=(50, 3)))
    def test_classic_generate_validator_correct_random(self, params: list[float]) -> None:
        NormalMeanVarianceMixtures._classic_generate_params_validation(params)

    @pytest.mark.parametrize(
        "params",
        [
            [],
            [1.111],
            [1, 1, 1],
            [1, 1, 1, 1],
            np.random.uniform(-100, 100, size=(100, 1)),
            np.random.uniform(-100, 100, size=(100, 100)),
        ],
    )
    def test_canonical_generate_validator_value_error_length(self, params: list[float]) -> None:
        with pytest.raises(ValueError, match="Expected 2 parameters"):
            NormalMeanVarianceMixtures._canonical_generate_params_validation(params)

    @pytest.mark.parametrize(
        "params",
        [
            [sys.float_info.max, sys.float_info.max],
            [sys.float_info.min, sys.float_info.min],
            [1, 1],
            [0, 0],
            [-1, -1],
            [0.333, 0.666],
        ],
    )
    def test_canonical_generate_validator_correct(self, params: list[float]) -> None:
        NormalMeanVarianceMixtures._canonical_generate_params_validation(params)

    @pytest.mark.parametrize("params", np.random.uniform(-100, 100, size=(50, 2)))
    def test_canonical_generate_validator_correct_random(self, params: list[float]) -> None:
        NormalMeanVarianceMixtures._canonical_generate_params_validation(params)
