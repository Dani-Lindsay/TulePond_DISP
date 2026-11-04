"""
renewal.py: renewal-model statistics for the next-event forecast (notebook 13).

With a few repeat intervals the parameters of a renewal model are poorly known, and a forecast from the best fit
alone is too narrow: it treats the fitted aperiodicity as exact (the lesson of Parkfield,
Jackson and Kagan 2006). The forecast here is the Bayesian posterior predictive with the open interval in the
likelihood (Nomura et al. 2011, eqs 4-7; Okada et al. 2012, eqs 2-4): parameter draws from the posterior given the
intervals, each giving a conditional forecast of the next onset, averaged with the open interval (no event between
the last onset and the last observation) in the likelihood. The priors are non-informative (below; for the lognormal
the Jeffreys prior, the LN-SST of Okada et al. 2012), not the informative aperiodicity prior that Nomura et al. (2011)
fitted to paleoseismic faults (mean about 0.37); notebook 13 section 7.1 checks the calibration.

Posteriors (all with priors that carry no information on the scale of the intervals):
- BPT (Brownian passage time, the inverse Gaussian with mean mu and shape lambda, aperiodicity alpha^2 = mu / lambda;
  Matthews et al. 2002). Prior uniform in the event rate 1/mu and in log lambda (p(mu, alpha) ~ mu^-2 alpha^-1). The
  posterior is then exact:
  lambda ~ Gamma((n - 1) / 2, rate R / 2) with R = sum(1/x - 1/xbar), and 1/mu | lambda ~ Normal(1/xbar,
  1 / (lambda n xbar)), truncated at 1/mu > 0. lambda R ~ chi2(n - 1) is the exact sampling pivot of the inverse
  Gaussian (Chhikara and Folks 1989), so the spread of alpha is the spread the data allow.
- BPT with alpha fixed (the Working Group branches): mu only, on a grid, with the same prior on 1/mu.
- Lognormal: log x ~ Normal(m, sigma^2), prior uniform in m and log sigma; exact (sigma^2 from a scaled inverse
  chi2(n - 1), m | sigma^2 normal).
- Weibull: log x ~ Gumbel(minimum) with location u = log(scale) and scale b = 1/shape, prior uniform in u and log b;
  on a grid.
- Poisson (the baseline): rate r with prior 1/r, so r ~ Gamma(n, T + e) given n intervals of total length T and the
  open interval e; the predictive is closed form.

Units: years throughout. Every function is deterministic for a given numpy Generator.
"""
import numpy as np
from scipy import stats


FAMILIES = ("bpt", "lognormal", "weibull")
MAX_EDGE_MASS = 1e-3            # grid posteriors: at most this posterior mass in the outermost grid cells


def bpt_distribution(mean_yr, alpha):
    """Brownian passage time (inverse-Gaussian) distribution with mean mean_yr and aperiodicity alpha.

    scipy's invgauss(mu=alpha^2, scale=mean/alpha^2) has mean mean_yr and standard deviation / mean alpha
    (Ellsworth et al. 1999, p3). Arrays broadcast.

    Parameters
    ----------
    mean_yr : float or numpy.ndarray
        Mean interval (yr).
    alpha : float or numpy.ndarray
        Aperiodicity.

    Returns
    -------
    scipy.stats frozen distribution
    """
    alpha = np.asarray(alpha, float)
    return stats.invgauss(mu=alpha ** 2, scale=np.asarray(mean_yr, float) / alpha ** 2)


def bpt_mle_alpha(intervals):
    """Maximum-likelihood aperiodicity of the BPT, alpha^2 = xbar * mean(1/x - 1/xbar) (Matthews et al. 2002).

    Parameters
    ----------
    intervals : array-like
        Repeat intervals (yr).

    Returns
    -------
    float
    """
    x = np.asarray(intervals, float)
    return float(np.sqrt(x.mean() * np.mean(1 / x - 1 / x.mean())))


def _sample_grid(log_post, rng, size):
    """Draw cell indices from an unnormalised log posterior on a grid; assert the mass at the grid edge is small.

    Parameters
    ----------
    log_post : numpy.ndarray
        Log posterior on the grid (1-D or 2-D).
    rng : numpy.random.Generator
    size : int

    Returns
    -------
    tuple of numpy.ndarray
        The drawn indices along each axis.
    """
    weights = np.exp(log_post - log_post.max())
    weights /= weights.sum()
    edge = np.zeros_like(weights, bool)
    for axis in range(weights.ndim):
        index = [slice(None)] * weights.ndim
        index[axis] = 0
        edge[tuple(index)] = True
        index[axis] = -1
        edge[tuple(index)] = True
    assert weights[edge].sum() < MAX_EDGE_MASS, f"posterior mass {weights[edge].sum():.2e} at the edge of the grid"
    flat = rng.choice(weights.size, size=size, p=weights.ravel())
    return np.unravel_index(flat, weights.shape)


def posterior_draws(model, intervals, rng, size, alpha_fixed=None):
    """Draws of the renewal-model parameters from their posterior given the intervals (module docstring).

    Parameters
    ----------
    model : str
        'bpt', 'lognormal' or 'weibull'.
    intervals : array-like
        Repeat intervals (yr), at least 2.
    rng : numpy.random.Generator
    size : int
        Number of draws.
    alpha_fixed : float, optional
        Fixed BPT aperiodicity (the Working Group branches): only the mean is drawn.

    Returns
    -------
    dist : scipy.stats frozen distribution
        The interval distributions of the draws, parameters of shape (size, 1), so that dist.cdf(t) broadcasts over a
        time axis of shape (n_times,).
    mean_yr : numpy.ndarray
        Mean interval of each draw (yr).
    cov : numpy.ndarray
        Standard deviation / mean of each draw (for the BPT, its aperiodicity).
    """
    x = np.asarray(intervals, float)
    n = len(x)
    assert n >= 2 and np.all(x > 0), f"need at least 2 positive intervals, got {x}"
    assert alpha_fixed is None or model == "bpt", "alpha_fixed is a BPT setting"
    if model == "bpt" and alpha_fixed is None:
        xbar = x.mean()
        r = np.sum(1 / x - 1 / xbar)                        # = n / lambda_hat
        lam = rng.gamma((n - 1) / 2, 2 / r, size)
        rate = rng.normal(1 / xbar, 1 / np.sqrt(lam * n * xbar))
        while (bad := rate <= 0).any():                     # the truncation at 1/mu > 0 (only for very wide posteriors)
            rate[bad] = rng.normal(1 / xbar, 1 / np.sqrt(lam[bad] * n * xbar))
        mean_yr = 1 / rate
        cov = np.sqrt(mean_yr / lam)
        return bpt_distribution(mean_yr[:, None], cov[:, None]), mean_yr, cov
    if model == "bpt":
        # mu on a log grid; the prior uniform in 1/mu is 1/mu per unit log mu
        log_mu = np.log(x.mean()) + np.linspace(-3, 3, 3001)
        mu = np.exp(log_mu)
        lam = mu / alpha_fixed ** 2
        log_lik = (0.5 * np.log(lam)[:, None] - lam[:, None] * (x - mu[:, None]) ** 2 / (2 * mu[:, None] ** 2 * x)).sum(axis=1)
        (k,) = _sample_grid(log_lik - log_mu, rng, size)
        step = log_mu[1] - log_mu[0]
        mean_yr = np.exp(log_mu[k] + rng.uniform(-step / 2, step / 2, size))
        cov = np.full(size, float(alpha_fixed))
        return bpt_distribution(mean_yr[:, None], cov[:, None]), mean_yr, cov
    if model == "lognormal":
        y = np.log(x)
        sigma2 = (n - 1) * y.var(ddof=1) / rng.chisquare(n - 1, size)
        m = rng.normal(y.mean(), np.sqrt(sigma2 / n))
        dist = stats.lognorm(s=np.sqrt(sigma2)[:, None], scale=np.exp(m)[:, None])
        return dist, np.exp(m + sigma2 / 2), np.sqrt(np.exp(sigma2) - 1)
    if model == "weibull":
        # log x ~ Gumbel(min), location u, scale b; grid in (log b, v) with u = ybar + gamma b + v b, prior 1/b;
        # the Jacobians (b for log b, b for v) and the prior give log posterior = log likelihood + log b
        y = np.log(x)
        b0 = max(y.std(ddof=1), 1e-3) * np.sqrt(6) / np.pi
        log_b = np.log(b0) + np.linspace(-4, 4, 161)
        v = np.linspace(-6, 6, 161)
        b = np.exp(log_b)[:, None]
        u = y.mean() + np.euler_gamma * b + v[None, :] * b
        z = (y[:, None, None] - u[None]) / b[None]
        log_post = (z - np.exp(z)).sum(axis=0) - n * np.log(b) + np.log(b)
        i, j = _sample_grid(log_post, rng, size)
        b_draw = np.exp(log_b[i] + rng.uniform(-0.5, 0.5, size) * (log_b[1] - log_b[0]))
        u_draw = y.mean() + np.euler_gamma * b_draw + (v[j] + rng.uniform(-0.5, 0.5, size) * (v[1] - v[0])) * b_draw
        dist = stats.weibull_min(c=(1 / b_draw)[:, None], scale=np.exp(u_draw)[:, None])
        return dist, dist.mean().ravel(), (dist.std() / dist.mean()).ravel()
    raise ValueError(model)


def predictive_terms(dist, t_last, elapsed, times):
    """The two sums of the posterior predictive of the next onset, given no onset for `elapsed` after `t_last`.

    For draws s with interval cdf F_s and survival S_s, the predictive probability of the next onset by time t is

        P(t) = sum_s [F_s(t - t_last) - F_s(elapsed)]+ / sum_s S_s(elapsed),

    the conditional forecast of each draw weighted by the probability it gives the quiet time (the open interval in
    the likelihood). Callers add up the numerator and the denominator over all their draws and divide once.

    Parameters
    ----------
    dist : scipy.stats frozen distribution
        Interval distributions with parameters of shape (S, 1) (posterior_draws).
    t_last : float or numpy.ndarray
        Last onset (decimal year), scalar or shape (S, 1).
    elapsed : float or numpy.ndarray
        Quiet time from the last onset to the reference date (yr), scalar or shape (S, 1).
    times : numpy.ndarray
        Times at which the predictive is wanted (decimal years), shape (n_times,).

    Returns
    -------
    numerator : numpy.ndarray
        Shape (n_times,).
    denominator : float
    """
    f_elapsed = dist.cdf(elapsed)
    numerator = np.clip(dist.cdf(times[None, :] - t_last) - f_elapsed, 0, None).sum(axis=0)
    return numerator, float(np.sum(1 - f_elapsed))


def poisson_predictive_cdf(intervals, elapsed, times, ref_yr):
    """Predictive probability of the next onset by each time under a Poisson process (the baseline).

    The rate r has prior 1/r; given n intervals of total length T and the open interval e (no event for e after the
    last onset), r ~ Gamma(n, T + e), and P(onset by t) = 1 - ((T + e) / (T + e + t - ref)) ** n for t > ref.

    Parameters
    ----------
    intervals : array-like
        Repeat intervals (yr).
    elapsed : float
        Open interval at the reference date (yr).
    times : numpy.ndarray
        Decimal years.
    ref_yr : float
        Reference date, the last observation (decimal year).

    Returns
    -------
    numpy.ndarray
    """
    x = np.asarray(intervals, float)
    exposure = x.sum() + elapsed
    ahead = np.clip(np.asarray(times, float) - ref_yr, 0, None)
    return 1 - (exposure / (exposure + ahead)) ** len(x)


def coverage(true_dist, n_intervals, elapsed_frac, rng, n_trials=500, n_post=400, model="bpt", levels=(0.05, 0.95, 0.99)):
    """Calibration of the posterior-predictive forecast on synthetic records with a known interval distribution.

    Each trial draws n_intervals intervals and a next interval from true_dist, keeps the trial only if the next
    interval exceeds the quiet time e = elapsed_frac x the true mean (as for the real forecast, issued after e of
    quiet), forecasts the next onset from the intervals and e, and records where the true next onset falls.

    Parameters
    ----------
    true_dist : scipy.stats frozen distribution
        The true interval distribution.
    n_intervals : int
        Intervals per synthetic record.
    elapsed_frac : float
        Quiet time at the forecast as a fraction of the true mean interval.
    rng : numpy.random.Generator
    n_trials : int
        Number of synthetic records.
    n_post : int
        Posterior draws per record.
    model : str
        The model the forecast fits.
    levels : tuple of float
        Quantile levels of the forecast.

    Returns
    -------
    dict
        For each level q, the fraction of trials whose next onset fell after the q quantile ('after_q05' and so on),
        and 'inside_05_95', the fraction inside the 5 to 95 % window.
    """
    mean_true = float(true_dist.mean())
    elapsed = elapsed_frac * mean_true
    times = elapsed + np.linspace(0, 8 * mean_true, 4001)       # time since the last onset
    after = {q: 0 for q in levels}
    inside = 0
    trials = 0
    while trials < n_trials:
        x = true_dist.rvs(size=n_intervals, random_state=rng)
        nxt = float(true_dist.rvs(random_state=rng))
        if nxt <= elapsed:
            continue
        dist, _, _ = posterior_draws(model, x, rng, n_post)
        numerator, denominator = predictive_terms(dist, 0.0, elapsed, times)
        cdf = numerator / denominator
        quantile = {q: times[min(int(np.searchsorted(cdf, q)), len(times) - 1)] for q in set(levels) | {0.05, 0.95}}
        for q in levels:
            after[q] += nxt > quantile[q]
        inside += quantile[0.05] <= nxt <= quantile[0.95]
        trials += 1
    out = {f"after_q{int(round(100 * q)):02d}": after[q] / n_trials for q in levels}
    out["inside_05_95"] = inside / n_trials
    return out
