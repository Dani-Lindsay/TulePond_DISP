"""renewal.py: the posteriors recover known parameters, the BPT posterior matches the exact inverse-Gaussian pivot,
the predictive and the Poisson baseline reach their known limits, and the forecast method is calibrated on synthetic
records (its 95 % date is passed without an event about 5 % of the time)."""
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import renewal                                                                  # noqa: E402

INSAR_INTERVALS = np.array([2.08, 1.92, 1.98, 1.91])                            # the four InSAR intervals (yr), rounded
MERGED_INTERVALS = np.array([2.85, 2.28, 4.13, 2.75, 2.80, 2.14, 0.88, 2.08, 1.92, 1.98, 1.91])


def test_bpt_posterior_recovers_known_parameters():
    rng = np.random.default_rng(1)
    x = renewal.bpt_distribution(2.0, 0.2).rvs(400, random_state=rng)
    _, mean, alpha = renewal.posterior_draws("bpt", x, rng, 4000)
    assert abs(np.median(mean) - 2.0) < 0.05
    assert abs(np.median(alpha) - 0.2) < 0.02


def test_bpt_posterior_matches_the_inverse_gaussian_pivot():
    # lambda R ~ chi2(n - 1): the posterior spread of the shape is the exact sampling spread (Chhikara and Folks 1989)
    rng = np.random.default_rng(2)
    _, mean, alpha = renewal.posterior_draws("bpt", INSAR_INTERVALS, rng, 20000)
    r = np.sum(1 / INSAR_INTERVALS - 1 / INSAR_INTERVALS.mean())
    assert stats.kstest(mean / alpha ** 2 * r, stats.chi2(len(INSAR_INTERVALS) - 1).cdf).pvalue > 0.01


def test_bpt_mle_alpha_formula():
    x = renewal.bpt_distribution(2.0, 0.3).rvs(20000, random_state=np.random.default_rng(3))
    assert abs(renewal.bpt_mle_alpha(x) - 0.3) < 0.01


def test_fixed_alpha_posterior_centres_on_the_sample_mean():
    # regression: a log-likelihood that counted 0.5 log(lambda) n times per interval put the mean at 2.7 x the sample mean
    rng = np.random.default_rng(4)
    for alpha in (0.3, 0.5, 0.7):
        _, mean, cov = renewal.posterior_draws("bpt", MERGED_INTERVALS, rng, 5000, alpha_fixed=alpha)
        assert np.all(cov == alpha)
        assert abs(np.median(mean) / MERGED_INTERVALS.mean() - 1) < 0.15


def test_lognormal_and_weibull_posteriors_recover_known_parameters():
    rng = np.random.default_rng(5)
    x = stats.lognorm(s=0.3, scale=2.0).rvs(400, random_state=rng)
    dist, _, _ = renewal.posterior_draws("lognormal", x, rng, 4000)
    assert abs(np.median(dist.kwds["s"]) - 0.3) < 0.03 and abs(np.median(dist.kwds["scale"]) - 2.0) < 0.08
    x = stats.weibull_min(c=4.0, scale=2.0).rvs(400, random_state=rng)
    dist, _, _ = renewal.posterior_draws("weibull", x, rng, 4000)
    assert abs(np.median(dist.kwds["c"]) - 4.0) < 0.4 and abs(np.median(dist.kwds["scale"]) - 2.0) < 0.05


def test_predictive_is_the_plug_in_forecast_when_the_parameters_are_known():
    rng = np.random.default_rng(6)
    true = renewal.bpt_distribution(2.0, 0.3)
    x = true.rvs(5000, random_state=rng)
    times = np.linspace(0, 8, 801)
    dist, _, _ = renewal.posterior_draws("bpt", x, rng, 2000)
    numerator, denominator = renewal.predictive_terms(dist, 0.0, 1.5, times)
    plug_in = np.clip((true.cdf(times) - true.cdf(1.5)) / true.sf(1.5), 0, 1)
    assert np.max(np.abs(numerator / denominator - plug_in)) < 0.02
    assert np.all(numerator[times <= 1.5] == 0)                                 # no onset before the reference date


def test_poisson_baseline_limits():
    times = np.linspace(0, 10, 101)
    p = renewal.poisson_predictive_cdf(np.full(20000, 2.0), 0.0, times, 0.0)    # rate known: 1 - exp(-t / 2)
    assert np.max(np.abs(p - (1 - np.exp(-times / 2.0)))) < 1e-3
    p = renewal.poisson_predictive_cdf(INSAR_INTERVALS, 1.8, times + 5.0, 5.0)
    assert p[0] == 0 and np.all(np.diff(p) >= 0) and p[-1] < 1


def test_forecast_is_calibrated_on_synthetic_records():
    # the InSAR case: 4 intervals, forecast issued 0.92 mean intervals after the last onset
    out = renewal.coverage(renewal.bpt_distribution(1.0, 0.1), 4, 0.92, np.random.default_rng(7), n_trials=200, n_post=200)
    assert 0.0 < out["after_q95"] <= 0.10
    assert 0.80 <= out["inside_05_95"] <= 0.97
