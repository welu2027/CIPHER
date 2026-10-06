"""Statistics helpers for the re-analysis (model-level correlations, bootstrap, paired tests)."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from scipy import stats

SEED = 42
N_BOOT = 10_000


def corr(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {"n": len(x), "pearson_r": float("nan"), "pearson_p": float("nan"),
                "spearman_rho": float("nan"), "spearman_p": float("nan")}
    pr = stats.pearsonr(x, y)
    sr = stats.spearmanr(x, y)
    return {"n": len(x), "pearson_r": float(pr.statistic), "pearson_p": float(pr.pvalue),
            "spearman_rho": float(sr.statistic), "spearman_p": float(sr.pvalue)}


def pearson_fisher_ci(r: float, n: int, alpha: float = 0.05) -> tuple:
    if n <= 3 or not np.isfinite(r) or abs(r) >= 1:
        return (float("nan"), float("nan"))
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    q = stats.norm.ppf(1 - alpha / 2)
    return (float(np.tanh(z - q * se)), float(np.tanh(z + q * se)))


def bootstrap_corr_ci(x, y, n_boot: int = N_BOOT, seed: int = SEED) -> Dict[str, float]:
    """Percentile bootstrap over models (resampling model indices with replacement)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    n = len(x)
    rs, rhos = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        xs, ys = x[idx], y[idx]
        if np.std(xs) == 0 or np.std(ys) == 0:
            continue
        rs.append(np.corrcoef(xs, ys)[0, 1])
        rhos.append(stats.spearmanr(xs, ys).statistic)
    rs, rhos = np.array(rs), np.array(rhos)
    return {"boot_valid": len(rs),
            "pearson_ci_lo": float(np.percentile(rs, 2.5)), "pearson_ci_hi": float(np.percentile(rs, 97.5)),
            "spearman_ci_lo": float(np.nanpercentile(rhos, 2.5)), "spearman_ci_hi": float(np.nanpercentile(rhos, 97.5))}


def paired_compare(a: Sequence[float], b: Sequence[float], n_boot: int = N_BOOT, seed: int = SEED) -> Dict[str, float]:
    """a, b aligned by instance. Bootstrap CI of mean(a-b), d_z, win/tie/loss and
    a two-sided exact sign test that DROPS ties (standard convention)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    rng = np.random.default_rng(seed)
    n = len(d)
    boots = d[rng.integers(0, n, (n_boot, n))].mean(axis=1)
    sd = d.std(ddof=1)
    eps = 1e-12
    wins, ties, losses = int((d > eps).sum()), int((np.abs(d) <= eps).sum()), int((d < -eps).sum())
    nz = wins + losses
    p_sign = float(stats.binomtest(wins, nz, 0.5).pvalue) if nz else float("nan")
    return {"n_pairs": n, "delta": float(d.mean()),
            "ci_lo": float(np.percentile(boots, 2.5)), "ci_hi": float(np.percentile(boots, 97.5)),
            "d_z": float(d.mean() / sd) if sd > 0 else float("nan"),
            "win_frac": wins / n, "tie_frac": ties / n, "loss_frac": losses / n,
            "win_frac_excl_ties": wins / nz if nz else float("nan"),
            "sign_p_ties_dropped": p_sign}


def kendall_tau(x, y) -> Dict[str, float]:
    t = stats.kendalltau(x, y)
    return {"tau": float(t.statistic), "p": float(t.pvalue)}


def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    if p < 1e-3:
        return f"{p:.1e}"
    return f"{p:.3f}"
