import numpy as np
import pytest
from scipy import ndimage
from scipy.stats import norm

from src.mixtures.nm_mixture import *


class TestMixingNormal:
    test_mixture_size = 100000

    @pytest.mark.parametrize(
        "mixing_variance, expected_variance", [(0, 1), (1, 2), (100, 101), (1.5, 2.5), (0.333, 1.333)]
    )
    def test_classic_generate_variance_0(self, mixing_variance: float, expected_variance: float) -> None:
        mixture = NormalMeanMixtures().classic_generate(self.test_mixture_size, norm, [0, mixing_variance**0.5, 1])
        actual_variance = ndimage.variance(mixture)
        assert actual_variance == pytest.approx(expected_variance, 0.1)

    @pytest.mark.parametrize("beta", np.random.uniform(0, 100, size=50))
    def test_classic_generate_variance_1(self, beta: float) -> None:
        expected_variance = beta**2 + 1
        mixture = NormalMeanMixtures().classic_generate(self.test_mixture_size, norm, [0, beta, 1])
        actual_variance = ndimage.variance(mixture)
        assert actual_variance == pytest.approx(expected_variance, 0.1)

    @pytest.mark.parametrize("beta, gamma", np.random.uniform(0, 100, size=(50, 2)))
    def test_classic_generate_variance_2(self, beta: float, gamma: float) -> None:
        expected_variance = beta**2 + gamma**2
        mixture = NormalMeanMixtures().classic_generate(self.test_mixture_size, norm, [0, beta, gamma])
        actual_variance = ndimage.variance(mixture)
        assert actual_variance == pytest.approx(expected_variance, 0.1)

    @pytest.mark.parametrize("beta, gamma", np.random.uniform(0, 10, size=(50, 2)))
    def test_classic_generate_mean(self, beta: float, gamma: float) -> None:
        expected_mean = 0
        mixture = NormalMeanMixtures().classic_generate(self.test_mixture_size, norm, [0, beta, gamma])
        actual_mean = np.mean(np.array(mixture))
        assert abs(actual_mean - expected_mean) < 1

    @pytest.mark.parametrize("expected_size", np.random.randint(0, 100, size=50))
    def test_classic_generate_size(self, expected_size: int) -> None:
        mixture = NormalMeanMixtures().classic_generate(expected_size, norm, [0, 1, 1])
        actual_size = np.size(mixture)
        assert actual_size == expected_size

    @pytest.mark.parametrize(
        "mixing_variance, expected_variance", [(0, 1), (1, 2), (100, 101), (1.5, 2.5), (0.333, 1.333)]
    )
    def test_canonical_generate_variance_0(self, mixing_variance: float, expected_variance: float) -> None:
        mixture = NormalMeanMixtures().canonical_generate(self.test_mixture_size, norm(0, mixing_variance**0.5), [1])
        actual_variance = ndimage.variance(mixture)
        assert actual_variance == pytest.approx(expected_variance, 0.1)

    @pytest.mark.parametrize("sigma", np.random.uniform(0, 100, size=50))
    def test_canonical_generate_variance_1(self, sigma: float) -> None:
        expected_variance = sigma**2 + 1
        mixture = NormalMeanMixtures().canonical_generate(self.test_mixture_size, norm, [sigma])
        actual_variance = ndimage.variance(mixture)
        assert actual_variance == pytest.approx(expected_variance, 0.1)

    @pytest.mark.parametrize("mixing_variance, sigma", np.random.uniform(0, 100, size=(50, 2)))
    def test_canonical_generate_variance_2(self, mixing_variance: float, sigma: float) -> None:
        expected_variance = mixing_variance + sigma**2
        mixture = NormalMeanMixtures().canonical_generate(
            self.test_mixture_size, norm(0, mixing_variance**0.5), [sigma]
        )
        actual_variance = ndimage.variance(mixture)
        assert actual_variance == pytest.approx(expected_variance, 0.1)

    @pytest.mark.parametrize("sigma", np.random.uniform(0, 10, size=50))
    def test_canonical_generate_mean(self, sigma: float) -> None:
        expected_mean = 0
        mixture = NormalMeanMixtures().canonical_generate(self.test_mixture_size, norm, [sigma])
        actual_mean = np.mean(np.array(mixture))
        assert abs(actual_mean - expected_mean) < 1

    @pytest.mark.parametrize("expected_size", [*np.random.randint(0, 100, size=50), 0, 1, 1000000])
    def test_canonical_generate_size(self, expected_size: int) -> None:
        mixture = NormalMeanMixtures().canonical_generate(expected_size, norm, [1])
        actual_size = np.size(mixture)
        assert actual_size == expected_size
