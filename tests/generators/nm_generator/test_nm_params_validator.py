import sys

import numpy as np
import pytest
from scipy.stats import norm

from src.mixtures.nm_mixture import NormalMeanMixtures


class TestGenerateParamsValidators:

    @pytest.mark.parametrize("params", [{}, {"a": 1, "b": 2}, {"a": 1, "b": 2, "c": 3, "d": 4, "m": 5}])
    def test_classical_args_validation_length_error(self, params):
        with pytest.raises(ValueError):
            NormalMeanMixtures._classical_params_validation(params)

    @pytest.mark.parametrize(
        "params",
        [
            {"wrong_alpha": 1, "beta": 1, "gamma": 1, "distribution": norm},
            {"alpha": 1, "wrong_beta": 1, "gamma": 1, "distribution": norm},
            {"alpha": 1, "beta": 1, "wrong_gamma": 1, "distribution": norm},
            {"alpha": 1, "beta": 1, "gamma": 1, "wrong_distribution": norm},
        ],
    )
    def test_classical_wrong_names(self, params):
        with pytest.raises(ValueError):
            NormalMeanMixtures._classical_params_validation(params)

    def test_classical_wrong_distribution_type(self):
        with pytest.raises(ValueError):
            NormalMeanMixtures._classical_params_validation({"alpha": 1, "beta": 1, "gamma": 1, "distribution": 1})

    @pytest.mark.parametrize("params", [{}, {"a": 1, "b": 2, "c": 3, "d": 4, "m": 5}])
    def test_canonical_args_validation_length_error(self, params):
        with pytest.raises(ValueError):
            NormalMeanMixtures._canonical_params_validation(params)

    @pytest.mark.parametrize(
        "params", [{"wrong_sigma": 1, "distribution": norm}, {"sigma": 1, "wrong_distribution": norm}]
    )
    def test_canonical_wrong_names(self, params):
        with pytest.raises(ValueError):
            NormalMeanMixtures._canonical_params_validation(params)

    def test_canonical_wrong_distribution_type(self):
        with pytest.raises(ValueError):
            NormalMeanMixtures._classical_params_validation({"sigma": 1, "distribution": 1})

    def test_canonical_wrong_sigma_sign(self):
        with pytest.raises(ValueError):
            NormalMeanMixtures._classical_params_validation({"sigma": -1, "distribution": norm})
