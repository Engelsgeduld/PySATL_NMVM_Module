import math

import pytest
from scipy.stats import expon, gamma, halfnorm, pareto

from src.mixtures.nmv_mixture import NormalMeanVarianceMixtures


class TestSemiParametricMuEstimation:
    @pytest.mark.parametrize("real_mu", [i for i in range(-3, 3)])
    def test_mu_estimation_expon_no_parameters(self, real_mu: float) -> None:
        mixture = NormalMeanVarianceMixtures()
        sample = mixture.canonical_generate(10000, expon, [0, real_mu])
        est_mu = mixture.semi_param_algorithm("mu_estimation", sample)
        assert abs(real_mu - est_mu) < 1

    @pytest.mark.parametrize("real_mu", [10**i for i in range(0, -10, -2)])
    def test_mu_estimation_expon_no_parameters_small(self, real_mu: float) -> None:
        mixture = NormalMeanVarianceMixtures()
        sample = mixture.canonical_generate(10000, expon, [0, real_mu])
        est_mu = mixture.semi_param_algorithm("mu_estimation", sample)
        assert abs(real_mu - est_mu) < 1

    @pytest.mark.parametrize("real_mu", [50, 100])
    def test_mu_estimation_expon_no_parameters_huge(self, real_mu: float) -> None:
        mixture = NormalMeanVarianceMixtures()
        sample = mixture.canonical_generate(1000000, expon, [0, real_mu])
        est_mu = mixture.semi_param_algorithm("mu_estimation", sample)
        assert abs(real_mu - est_mu) < real_mu / 2

    @pytest.mark.parametrize("real_mu", [i for i in range(-3, 3)])
    def test_mu_estimation_pareto_no_parameters(self, real_mu: float) -> None:
        mixture = NormalMeanVarianceMixtures()
        sample = mixture.canonical_generate(50000, pareto(2.62), [0, real_mu])
        est_mu = mixture.semi_param_algorithm("mu_estimation", sample)
        assert abs(real_mu - est_mu) < 1

    @pytest.mark.parametrize("real_mu", [10**i for i in range(0, -10, -2)])
    def test_mu_estimation_pareto_no_parameters_small(self, real_mu: float) -> None:
        mixture = NormalMeanVarianceMixtures()
        sample = mixture.canonical_generate(50000, pareto(2.62), [0, real_mu])
        est_mu = mixture.semi_param_algorithm("mu_estimation", sample)
        assert abs(real_mu - est_mu) < 1

    @pytest.mark.parametrize("real_mu", [i for i in range(-3, 3)])
    def test_mu_estimation_halfnorm_no_parameters(self, real_mu: float) -> None:
        mixture = NormalMeanVarianceMixtures()
        sample = mixture.canonical_generate(10000, halfnorm, [0, real_mu])
        est_mu = mixture.semi_param_algorithm("mu_estimation", sample)
        assert abs(real_mu - est_mu) < 1

    @pytest.mark.parametrize("real_mu", [i for i in range(-3, 3)])
    def test_mu_estimation_gamma_no_parameters(self, real_mu: float) -> None:
        mixture = NormalMeanVarianceMixtures()
        sample = mixture.canonical_generate(10000, gamma(2), [0, real_mu])
        est_mu = mixture.semi_param_algorithm("mu_estimation", sample)
        assert abs(real_mu - est_mu) < 1

    @pytest.mark.parametrize("real_mu, params", [[1, [m]] for m in range(5, 10)])
    def test_mu_estimation_expon_1_parameter_m_positive(self, real_mu: float, params: list) -> None:
        mixture = NormalMeanVarianceMixtures()
        sample = mixture.canonical_generate(10000, expon, [0, real_mu])
        est_mu = mixture.semi_param_algorithm("mu_estimation", sample, params)
        assert abs(real_mu - est_mu) < 1

    @pytest.mark.parametrize("real_mu, params", [[-1, [m]] for m in range(5, 10)])
    def test_mu_estimation_expon_1_parameter_m_negative(self, real_mu: float, params: list) -> None:
        mixture = NormalMeanVarianceMixtures()
        sample = mixture.canonical_generate(10000, expon, [0, real_mu])
        est_mu = mixture.semi_param_algorithm("mu_estimation", sample, params)
        assert abs(real_mu - est_mu) < 1

    @pytest.mark.parametrize("real_mu, params", [[10, [i]] for i in range(1, 10)])
    def test_mu_estimation_expon_1_parameter_m_is_best_estimation(self, real_mu: float, params: list) -> None:
        mixture = NormalMeanVarianceMixtures()
        sample = mixture.canonical_generate(10000, expon, [0, real_mu])
        est_mu = mixture.semi_param_algorithm("mu_estimation", sample, params)
        assert abs(est_mu == params[0])

    @pytest.mark.parametrize(
        "real_mu, params", [[1, [10, tol]] for tol in [1 / 10**6, 1 / 10**7, 1 / 10**8, 1 / 10**9, 1 / 10**10]]
    )
    def test_mu_estimation_expon_2_parameters_tol_positive(self, real_mu: float, params: list) -> None:
        mixture = NormalMeanVarianceMixtures()
        sample = mixture.canonical_generate(10000, expon, [0, real_mu])
        est_mu = mixture.semi_param_algorithm("mu_estimation", sample, params)
        assert abs(real_mu - est_mu) < 1

    @pytest.mark.parametrize(
        "real_mu, params", [[-1, [10, tol]] for tol in [1 / 10**6, 1 / 10**7, 1 / 10**8, 1 / 10**9, 1 / 10**10]]
    )
    def test_mu_estimation_expon_2_parameters_tol_negative(self, real_mu: float, params: list) -> None:
        mixture = NormalMeanVarianceMixtures()
        sample = mixture.canonical_generate(10000, expon, [0, real_mu])
        est_mu = mixture.semi_param_algorithm("mu_estimation", sample, params)
        assert abs(real_mu - est_mu) < 1

    @pytest.mark.parametrize(
        "real_mu, params",
        [
            [1, [10, 1 / 10**9, omega]]
            for omega in [
                lambda x: -1 * (x**i) * math.exp(-(1 / (1 - x**2))) if abs(x) < 1 else 0 for i in range(1, 20, 2)
            ]
        ],
    )
    def test_mu_estimation_expon_3_parameters_omega_positive(self, real_mu: float, params: list) -> None:
        mixture = NormalMeanVarianceMixtures()
        sample = mixture.canonical_generate(10000, expon, [0, real_mu])
        est_mu = mixture.semi_param_algorithm("mu_estimation", sample, params)
        assert abs(real_mu - est_mu) < 1

    @pytest.mark.parametrize(
        "real_mu, params",
        [
            [-1, [10, 1 / 10**9, omega]]
            for omega in [
                lambda x: -1 * (x**i) * math.exp(-(1 / (1 - x**2))) if abs(x) < 1 else 0 for i in range(1, 20, 2)
            ]
        ],
    )
    def test_mu_estimation_expon_3_parameters_omega_negative(self, real_mu: float, params: list) -> None:
        mixture = NormalMeanVarianceMixtures()
        sample = mixture.canonical_generate(10000, expon, [0, real_mu])
        est_mu = mixture.semi_param_algorithm("mu_estimation", sample, params)
        assert abs(real_mu - est_mu) < 1

    @pytest.mark.parametrize(
        "real_mu, params",
        [
            [1, [m, tol, omega]]
            for m in range(10, 100, 30)
            for tol in [1 / 10**6, 1 / 10**7, 1 / 10**8]
            for omega in [
                lambda x: -1 * (x**i) * math.exp(-(1 / (1 - x**2))) if abs(x) < 1 else 0 for i in range(1, 6, 2)
            ]
        ],
    )
    def test_mu_estimation_expon_3_parameters_all_positive(self, real_mu: float, params: list) -> None:
        mixture = NormalMeanVarianceMixtures()
        sample = mixture.canonical_generate(10000, expon, [0, real_mu])
        est_mu = mixture.semi_param_algorithm("mu_estimation", sample, params)
        assert abs(real_mu - est_mu) < 1

    @pytest.mark.parametrize(
        "real_mu, params",
        [
            [-1, [m, tol, omega]]
            for m in range(10, 100, 30)
            for tol in [1 / 10**6, 1 / 10**7, 1 / 10**8]
            for omega in [
                lambda x: -1 * (x**i) * math.exp(-(1 / (1 - x**2))) if abs(x) < 1 else 0 for i in range(1, 6, 2)
            ]
        ],
    )
    def test_mu_estimation_expon_3_parameters_all_negative(self, real_mu: float, params: list) -> None:
        mixture = NormalMeanVarianceMixtures()
        sample = mixture.canonical_generate(10000, expon, [0, real_mu])
        est_mu = mixture.semi_param_algorithm("mu_estimation", sample, params)
        assert abs(real_mu - est_mu) < 1
