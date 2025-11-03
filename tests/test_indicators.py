# path: tests/test_indicators.py
import os
import numpy as np
import pandas as pd

from project.hybrid_trader import (
    bulls_bears, triple_barrier_labels, threshold_sweep, backtest_long_only
)


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
    assert len(nb) == len(close)
    assert len(nr) == len(close)
    # valores dentro de [-100, 100] con tolerancia
    assert np.nanmax(nb.values) <= 100.0001
    assert np.nanmin(nb.values) >= -100.0001
    assert np.nanmax(nr.values) <= 100.0001
    assert np.nanmin(nr.values) >= -100.0001
    # total puede exceder ligeramente por borde, pero no debería ser enorme
    assert np.nanmax(np.abs(total.values)) < 205.0


def test_triple_barrier_labels_monotonic_up_down():
    n = 120
    idx = pd.date_range("2021-01-01", periods=n, freq="D")
    # Serie creciente suave
    close_up = pd.Series(100.0 * (1.003 ** np.arange(n)), index=idx)
    vol = pd.Series(0.001, index=idx)
    y_up = triple_barrier_labels(close_up, horizon=5, k=2.0, vol=vol)
    # Esperado: muchos 1 al principio, últimos 'horizon' forzados a 0
    assert y_up.iloc[:-5].mean() > 0.8
    assert y_up.iloc[-5:].sum() == 0

    # Serie decreciente
    close_dn = pd.Series(100.0 * (1.0 / (1.003 ** np.arange(n))), index=idx)
    y_dn = triple_barrier_labels(close_dn, horizon=5, k=2.0, vol=vol)
    # Esperado: mayormente 0
    assert y_dn.mean() < 0.2


def test_threshold_sweep_outputs(tmp_path):
    # Datos sintéticos
    n = 252
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    close = pd.Series(np.cumprod(1 + np.random.normal(0, 0.01, n)), index=idx)
    proba = pd.Series(np.clip(np.random.normal(0.5, 0.15, n), 0, 1), index=idx)

    outdir = tmp_path / "report"
    df = threshold_sweep(close, proba, fees_bps=5.0, slippage_bps=5.0, outdir=str(outdir))
    # 21 thresholds de 0.4 a 0.8
    assert len(df) == 21
    assert np.isclose(df["threshold"].iloc[0], 0.4)
    assert np.isclose(df["threshold"].iloc[-1], 0.8)
    # Columnas clave presentes
    for c in ["CAGR","Sharpe","MaxDD","HitRate","TurnsPerYear"]:
        assert c in df.columns
    # Archivos creados
    assert (outdir / "sweep_cagr.png").exists()
    assert (outdir / "sweep_sharpe.png").exists()
    assert (outdir / "threshold_sweep.csv").exists()
