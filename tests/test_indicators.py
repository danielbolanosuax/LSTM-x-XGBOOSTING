# path: tests/test_indicators.py
import numpy as np
import pandas as pd
from project.hybrid_trader import bulls_bears, triple_barrier_labels, threshold_sweep

def _mk_series(n=300, seed=123):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    close = pd.Series(np.cumsum(rng.normal(0, 0.5, n)) + 100.0, index=idx)
    high = close + np.abs(rng.normal(0.2, 0.1, n))
    low = close - np.abs(rng.normal(0.2, 0.1, n))
    return close, high, low

def test_bulls_bears_range_and_shape():
    close, high, low = _mk_series()
    total, nb, nr = bulls_bears(close, high, low, length=14, bars_back=60)
    assert len(total) == len(close)
    assert np.nanmax(nb.values) <= 100.0001 and np.nanmin(nb.values) >= -100.0001
    assert np.nanmax(nr.values) <= 100.0001 and np.nanmin(nr.values) >= -100.0001
    assert np.nanmax(np.abs(total.values)) < 205.0

def test_triple_barrier_labels_monotonic_up_down():
    n = 120
    idx = pd.date_range("2021-01-01", periods=n, freq="D")
    close_up = pd.Series(100.0 * (1.003 ** np.arange(n)), index=idx)
    vol = pd.Series(0.001, index=idx)
    y_up = triple_barrier_labels(close_up, horizon=5, k=2.0, vol=vol)
    assert y_up.iloc[:-5].mean() > 0.8 and y_up.iloc[-5:].sum() == 0
    close_dn = pd.Series(100.0 * (1.0 / (1.003 ** np.arange(n))), index=idx)
    y_dn = triple_barrier_labels(close_dn, horizon=5, k=2.0, vol=vol)
    assert y_dn.mean() < 0.2

def test_threshold_sweep_outputs(tmp_path):
    n = 252
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    close = pd.Series(np.cumprod(1 + np.random.normal(0, 0.01, n)), index=idx)
    proba = pd.Series(np.clip(np.random.normal(0.5, 0.15, n), 0, 1), index=idx)
    outdir = tmp_path / "report"
    df = threshold_sweep(close, proba, fees_bps=5.0, slippage_bps=5.0, outdir=str(outdir))
    assert len(df) == 21 and np.isclose(df["threshold"].iloc[0], 0.4) and np.isclose(df["threshold"].iloc[-1], 0.8)
    for c in ["CAGR","Sharpe","MaxDD","HitRate","TurnsPerYear"]:
        assert c in df.columns
