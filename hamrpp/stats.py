"""Paired significance tests, Holm correction and bootstrap confidence intervals (Sec. 5.2)."""
import math
import numpy as np
from scipy import stats


def wilcoxon_p(x, y):
    """Two-sided Wilcoxon signed-rank test on paired values; pairs with a NaN are dropped.
    Returns (p, n_pairs). p = 1 if all differences are zero."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) == 0 or np.allclose(x - y, 0):
        return 1.0, int(len(x))
    return float(stats.wilcoxon(x, y, alternative="two-sided").pvalue), int(len(x))


def mcnemar_exact_p(a, b):
    """McNemar's exact test on paired binary outcomes (two-sided binomial test on discordant pairs)."""
    a, b = np.asarray(a, int), np.asarray(b, int)
    n10 = int(((a == 1) & (b == 0)).sum())
    n01 = int(((a == 0) & (b == 1)).sum())
    if n10 + n01 == 0:
        return 1.0, (n10, n01)
    return float(stats.binomtest(n10, n10 + n01, 0.5, alternative="two-sided").pvalue), (n10, n01)


def holm(pvals):
    """Holm step-down adjusted p values (same order as input)."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


def bootstrap_ci(values_by_trial, stat=np.nanmean, n_boot=2000, seed=12345, alpha=0.05):
    """Nonparametric bootstrap over trials. `values_by_trial` is a list (one entry per trial) of
    arrays (e.g. per-robot values); the statistic is computed on the pooled resample."""
    rng = np.random.default_rng(seed)
    groups = [np.asarray(v, float) for v in values_by_trial]
    n = len(groups)
    reps = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        pooled = np.concatenate([groups[i] for i in idx]) if n else np.array([])
        reps.append(stat(pooled) if pooled.size else math.nan)
    lo, hi = np.nanpercentile(reps, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)
