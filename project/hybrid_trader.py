# path: project/hybrid_trader.py
from __future__ import annotations

import os
import sys
import json
import time
import math
import argparse
import pathlib
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd

# Evita problemas de SSL con yfinance/curl_cffi en Windows
os.environ.setdefault("YFINANCE_USE_CURL_CFFI", "0")

# ====== Imports opcionales (no rompemos si faltan) ======
try:
    from alpha_vantage.timeseries import TimeSeries  # type: ignore
except Exception:
    TimeSeries = None

# Import perezoso de yfinance (lo reintentamos dentro de fetch)
try:
    import yfinance as _yf  # type: ignore
except Exception:
    _yf = None

try:
    from xgboost import XGBClassifier  # type: ignore
except Exception:
    XGBClassifier = None

# reporting (intenta importar como paquete o módulo local)
try:
    ROOT = pathlib.Path(__file__).resolve().parent.parent
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
except Exception:
    pass

try:
    from project import reporting as _reporting  # type: ignore
except Exception:
    try:
        import reporting as _reporting  # type: ignore
    except Exception:
        _reporting = None


# ====== Utilidades de logging/progreso ======
def _now() -> str:
    return time.strftime("%H:%M:%S")

def log_step(i: int, n: int, msg: str) -> float:
    print(f"[{_now()}] [{i}/{n}] {msg}", flush=True)
    return time.time()

def log_ok(t0: float, extra: str = "") -> None:
    dt = time.time() - t0
    s = f"OK ({dt:.1f}s)"
    if extra:
        s += f" - {extra}"
    print(f"    {s}", flush=True)

def log_warn(msg: str) -> None:
    print(f"    [WARN] {msg}", flush=True)

def log_err(msg: str) -> None:
    print(f"    [ERROR] {msg}", flush=True)


# ====== Config ======
@dataclass
class Config:
    ticker: str = "AAPL"
    start_date: str = "2014-01-01"
    end_date: Optional[str] = None
    window: int = 60
    horizon: int = 20
    barrier_mult: float = 2.0
    scaler_type: str = "standard"
    test_splits: int = 5
    gap: int = 3
    fees_bps: float = 5.0
    slippage_bps: float = 5.0
    proba_threshold: float = 0.6
    seed: int = 42
    rl_lambda_risk: float = 0.0
    rl_steps: int = 200_000
    action_smooth_tau: float = 0.2
    max_pos_change: float = 0.3
    vol_target: float = 0.2
    # Data source
    alpha_vantage_key_env: str = "ALPHA_VANTAGE_KEY"
    av_key: Optional[str] = None
    data_source: str = "auto"   # auto | av | yahoo | stooq
    use_adjusted_close: bool = True
    # BvB
    bvb_len: int = 14
    bvb_bars_back: int = 120
    bvb_tline: float = 80.0
    plot_bvb: bool = False
    # Verbose / cache
    verbose_train: bool = True
    cache_dir: str = "data"


# ====== Paths / Cache (CSV, sin pyarrow) ======
def _cache_key(cfg: Config) -> str:
    sd = cfg.start_date.replace("-", "")
    ed = (cfg.end_date or "None").replace("-", "")
    adj = "adj" if cfg.use_adjusted_close else "raw"
    return f"{cfg.ticker}_{sd}_{ed}_{adj}.csv"

def _cache_path(cfg: Config) -> pathlib.Path:
    pathlib.Path(cfg.cache_dir).mkdir(parents=True, exist_ok=True)
    return pathlib.Path(cfg.cache_dir) / _cache_key(cfg)

def _artifacts_dir() -> pathlib.Path:
    d = pathlib.Path("artifacts")
    d.mkdir(parents=True, exist_ok=True)
    return d

def _read_cache_csv(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df

def _write_cache_csv(df: pd.DataFrame, path: pathlib.Path) -> None:
    df.to_csv(path)


# ====== Descarga desde Stooq (fallback sin dependencias) ======
def _fetch_stooq(cfg: Config) -> pd.DataFrame:
    sym = cfg.ticker.lower()
    if "." not in sym and sym.isalpha():
        sym = sym + ".us"
    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    print(f"    [Stooq] GET {url}", flush=True)
    df = pd.read_csv(url)
    if df.empty:
        raise RuntimeError("Stooq devolvió vacío")
    df.columns = [c.title() for c in df.columns]
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
    cols_map = {"Open": "Open", "High": "High", "Low": "Low", "Close": "Close", "Volume": "Volume"}
    df = df.rename(columns=cols_map)[["Open", "High", "Low", "Close", "Volume"]]
    df = df.sort_index()
    df = df[df.index >= pd.to_datetime(cfg.start_date)]
    if cfg.end_date:
        df = df[df.index <= pd.to_datetime(cfg.end_date)]
    df = df.dropna()
    if df.empty:
        raise RuntimeError("Stooq: sin datos tras limpieza")
    return df


# ====== Fetch OHLCV con fallback AV -> Yahoo -> Stooq ======
def fetch_ohlcv(cfg: Config) -> pd.DataFrame:
    cp = _cache_path(cfg)
    if cp.exists():
        t0 = log_step(1, 6, f"Cargando cache {cp.name}")
        df = _read_cache_csv(cp)
        log_ok(t0, f"{len(df)} barras")
        return df

    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df[df.index >= pd.to_datetime(cfg.start_date)]
        if cfg.end_date:
            df = df[df.index <= pd.to_datetime(cfg.end_date)]
        need = ["Open", "High", "Low", "Close", "Volume"]
        for c in need:
            if c not in df.columns:
                raise RuntimeError(f"Falta columna {c}")
        df = df[need].dropna()
        if df.empty:
            raise RuntimeError("Sin datos tras limpieza")
        return df

    def try_alpha_vantage() -> Optional[pd.DataFrame]:
        key = (cfg.av_key or os.getenv(cfg.alpha_vantage_key_env, "")).strip()
        if not key or TimeSeries is None:
            print("    [AV] KEY no disponible o lib no importable; salto.", flush=True)
            return None
        t0 = log_step(1, 6, f"Descargando Alpha Vantage ({cfg.ticker})")
        ts = TimeSeries(key=key, output_format="pandas")
        for step in ("adjusted", "daily"):
            try:
                if step == "adjusted" and cfg.use_adjusted_close:
                    data, _ = ts.get_daily_adjusted(symbol=cfg.ticker, outputsize="full")
                    data = data.rename(columns={
                        "1. open": "Open", "2. high": "High", "3. low": "Low",
                        "4. close": "Close", "5. adjusted close": "Adj Close", "6. volume": "Volume"
                    })
                    if "Adj Close" in data.columns:
                        data["Close"] = data["Adj Close"]
                else:
                    data, _ = ts.get_daily(symbol=cfg.ticker, outputsize="full")
                    data = data.rename(columns={
                        "1. open": "Open", "2. high": "High", "3. low": "Low",
                        "4. close": "Close", "5. volume": "Volume"
                    })
                df = _clean(data)
                log_ok(t0, f"{len(df)} barras")
                return df
            except Exception as e:
                msg = str(e)
                log_warn(f"AV intento '{step}' falló: {msg}")
                if "Please consider" in msg or "exceeded" in msg.lower():
                    time.sleep(12)
        return None

    def try_yahoo() -> Optional[pd.DataFrame]:
        yf = _yf
        if yf is None:
            try:
                import yfinance as yf  # type: ignore
            except Exception as e:
                print(f"    [Yahoo] yfinance no importable; salto. ({e})", flush=True)
                return None
        t0 = log_step(1, 6, f"Descargando Yahoo ({cfg.ticker})")
        try:
            df = yf.download(
                cfg.ticker, start=cfg.start_date, end=cfg.end_date,
                auto_adjust=True, progress=False, threads=False,
            )
            if df is None or df.empty:
                raise RuntimeError("Yahoo devolvió vacío")
            df = df.rename(columns=str.title)[["Open", "High", "Low", "Close", "Volume"]]
            df = _clean(df)
            log_ok(t0, f"{len(df)} barras")
            return df
        except Exception as e:
            log_warn(f"Yahoo fallo: {e!r}")
            return None

    def try_stooq() -> Optional[pd.DataFrame]:
        try:
            t0 = log_step(1, 6, f"Descargando Stooq ({cfg.ticker})")
            df = _fetch_stooq(cfg)
            log_ok(t0, f"{len(df)} barras")
            return df
        except Exception as e:
            log_warn(f"Stooq fallo: {e!r}")
            return None

    src = cfg.data_source.lower()
    if src == "auto":
        order = ["av", "yahoo", "stooq"]
    elif src in ("av", "yahoo", "stooq"):
        order = [src]
    else:
        raise ValueError("data_source debe ser auto|av|yahoo|stooq")

    for provider in order:
        df = None
        if provider == "av":
            df = try_alpha_vantage()
        elif provider == "yahoo":
            df = try_yahoo()
        else:
            df = try_stooq()
        if df is not None:
            _write_cache_csv(df, _cache_path(cfg))
            print(f"    [OK] Proveedor usado: {provider.upper()}", flush=True)
            return df

    raise RuntimeError("No se pudieron obtener datos OHLCV (AV/Yahoo/Stooq).")


# ====== Indicadores ======
def rsi_ema(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = np.where(delta > 0, delta, 0.0)
    dn = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=close.index).ewm(span=n, adjust=False).mean()
    roll_dn = pd.Series(dn, index=close.index).ewm(span=n, adjust=False).mean()
    rs = roll_up / (roll_dn + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def stoch_kd(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    lowest_low = low.rolling(k_period, min_periods=1).min()
    highest_high = high.rolling(k_period, min_periods=1).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-12)
    d = k.rolling(d_period, min_periods=1).mean()
    return k, d

def bulls_bears_tv(high: pd.Series, low: pd.Series, close: pd.Series,
                   length: int = 14, bars_back: int = 120, tline: float = 80.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ma = close.ewm(span=length, adjust=False).mean()
    bulls = high - ma
    bears = ma - low

    min_bulls = bulls.rolling(bars_back, min_periods=1).min()
    max_bulls = bulls.rolling(bars_back, min_periods=1).max()
    norm_bulls = ((bulls - min_bulls) / (max_bulls - min_bulls + 1e-12) - 0.5) * 100.0

    min_bears = bears.rolling(bars_back, min_periods=1).min()
    max_bears = bears.rolling(bars_back, min_periods=1).max()
    norm_bears = ((bears - min_bears) / (max_bears - min_bears + 1e-12) - 0.5) * 100.0

    total = norm_bulls - norm_bears
    bullish = (total > float(tline)).astype(int)
    bearish = (total < -float(tline)).astype(int)
    total.name = "BvB_Total"
    bullish.name = "BvB_Bullish"
    bearish.name = "BvB_Bearish"
    return total, bullish, bearish


# ====== Features ======
def add_features(raw: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    t0 = log_step(2, 6, "Construyendo features")
    df = raw.copy()
    df["Return"] = df["Close"].pct_change().fillna(0.0)
    df["LogRet"] = np.log(df["Close"]).diff().fillna(0.0)

    df["RSI"] = rsi_ema(df["Close"], 14)
    macd_line, macd_sig, macd_hist = macd(df["Close"], 12, 26, 9)
    df["MACD"] = macd_line
    df["MACD_Signal"] = macd_sig
    df["MACD_Hist"] = macd_hist
    st_k, st_d = stoch_kd(df["High"], df["Low"], df["Close"], 14, 3)
    df["StochK"] = st_k
    df["StochD"] = st_d

    df["Volatility20"] = df["LogRet"].rolling(20, min_periods=5).std()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()

    bvb_total, bvb_bullish, bvb_bearish = bulls_bears_tv(
        df["High"], df["Low"], df["Close"],
        length=cfg.bvb_len, bars_back=cfg.bvb_bars_back, tline=cfg.bvb_tline
    )
    df["BvB_Total"] = bvb_total
    df["BvB_Bullish"] = bvb_bullish
    df["BvB_Bearish"] = bvb_bearish

    df = df.dropna().copy()
    log_ok(t0, f"{len(df)} filas")
    return df


# ====== Labels (triple barrier simplificado) ======
def triple_barrier_labels(close: pd.Series, horizon: int, k: float, vol: pd.Series) -> pd.Series:
    labels = np.zeros(len(close), dtype=int)
    c = close.values
    sig = pd.Series(vol).fillna(pd.Series(vol).median()).values
    up_mult = (1 + k * sig)
    dn_mult = (1 - k * sig)
    n = len(c)
    for i in range(n):
        t_end = min(i + horizon, n - 1)
        up = c[i] * up_mult[i]
        dn = c[i] * dn_mult[i]
        hit_up = False
        hit_dn = False
        for j in range(i + 1, t_end + 1):
            if c[j] >= up:
                hit_up = True
                break
            if c[j] <= dn:
                hit_dn = True
                break
        labels[i] = 1 if (hit_up and not hit_dn) else 0
    if horizon > 0:
        labels[-horizon:] = 0
    return pd.Series(labels, index=close.index, name="y")


# ====== Backtest sencillo long-only por probas ======
def compute_drawdown(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return dd

def backtest_long_only(close: pd.Series, proba: pd.Series, threshold: float, fees_bps: float, slippage_bps: float) -> Dict[str, float]:
    proba = proba.fillna(0.0)
    signal = (proba >= threshold).astype(int)
    pos = signal.shift(1).fillna(0)
    ret = close.pct_change().fillna(0.0)
    gross = pos * ret
    turns = pos.diff().abs().fillna(0.0)
    cost = turns * ((fees_bps + slippage_bps) / 1e4)
    net = gross - cost
    equity = (1 + net).cumprod()
    dd = compute_drawdown(equity)
    cagr = equity.iloc[-1] ** (252 / len(equity)) - 1.0 if len(equity) else 0.0
    sharpe = np.sqrt(252) * (net.mean() / (net.std() + 1e-12))
    return {
        "CAGR": float(cagr),
        "Sharpe": float(sharpe),
        "MaxDD": float(dd.min()),
        "HitRate": float((gross > 0).mean()),
        "TurnsPerYear": float(turns.sum() * (252 / len(turns)))
    }


# ====== Dataset y entrenamiento ======
FEATURES = [
    "Return", "LogRet",
    "RSI", "MACD", "MACD_Signal", "MACD_Hist", "StochK", "StochD",
    "Volatility20",
    "MA20", "MA50", "MA200",
    "BvB_Total", "BvB_Bullish", "BvB_Bearish"
]

def build_xy(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    t0 = log_step(3, 6, "Etiquetando (triple barrier)")
    y = triple_barrier_labels(df["Close"], horizon=cfg.horizon, k=cfg.barrier_mult, vol=df["Volatility20"])
    log_ok(t0)
    X = df[FEATURES].copy()
    idx = df.index
    y = y.reindex(idx)
    mask = (~X.isna().any(axis=1)) & (~y.isna())
    X = X[mask]
    y = y[mask]
    return X, y, idx[mask]

def time_series_oof_prob(X: pd.DataFrame, y: pd.Series, n_splits: int = 5, random_state: int = 42) -> Tuple[pd.Series, pd.Series]:
    """Devuelve proba OOF y número de fold por fila."""
    if XGBClassifier is None:
        raise RuntimeError("xgboost no está instalado.")
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import roc_auc_score, average_precision_score

    t0 = log_step(4, 6, f"Entrenando (TimeSeriesSplit={n_splits})")
    tscv = TimeSeriesSplit(n_splits=n_splits)
    proba = pd.Series(index=X.index, dtype=float, name="proba")
    fold_ser = pd.Series(index=X.index, dtype="Int64", name="fold")
    aucs, aps = [], []

    for fold, (tr, te) in enumerate(tscv.split(X), 1):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=400,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=random_state,
                n_jobs=0,
                tree_method="hist",
                eval_metric="logloss",
                verbosity=0
            ))
        ])
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xte)[:, 1]
        proba.iloc[te] = p
        fold_ser.iloc[te] = fold  # por análisis por split
        try:
            aucs.append(roc_auc_score(yte, p))
            aps.append(average_precision_score(yte, p))
        except Exception:
            pass

    extra = f"AUC mean={np.mean(aucs):.3f} AP mean={np.mean(aps):.3f}" if aucs and aps else ""
    log_ok(t0, extra)
    return proba, fold_ser


# ====== BvB export ======
def save_bvb(df: pd.DataFrame, outdir: pathlib.Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    df[["BvB_Total", "BvB_Bullish", "BvB_Bearish"]].to_csv(outdir / "bvb_series.csv")
    import matplotlib.pyplot as plt
    plt.figure()
    df["BvB_Total"].plot(title="BvB_Total")
    plt.xlabel("Fecha"); plt.ylabel("BvB_Total")
    plt.tight_layout(); plt.savefig(outdir / "bvb_total.png"); plt.close()

    plt.figure()
    (df["BvB_Bullish"] - df["BvB_Bearish"]).plot(title="BvB señales (+1 bull / -1 bear)")
    plt.xlabel("Fecha"); plt.ylabel("Señal")
    plt.tight_layout(); plt.savefig(outdir / "bvb_signals.png"); plt.close()


# ====== RUNS ======
def run_ml(cfg: Config) -> Tuple[pd.DataFrame, pd.Series]:
    print(f"Config: {cfg}", flush=True)

    # 1) datos
    raw = fetch_ohlcv(cfg)

    # 2) features
    df = add_features(raw, cfg)

    # 3) dataset + labels
    X, y, y_index = build_xy(df, cfg)

    # 4) entrenamiento OOF (guardamos fold)
    proba_oof, fold_oof = time_series_oof_prob(X, y, n_splits=cfg.test_splits, random_state=cfg.seed)

    # 5) artifacts
    t0 = log_step(5, 6, "Guardando artifacts")
    art = _artifacts_dir()

    # ====== GUARDAR PROBABILIDADES + ETIQUETA + FOLD ======
    dates = pd.to_datetime(X.index)
    assert len(dates) == len(proba_oof) == len(y) == len(fold_oof), "dates/proba/y_true/fold no cuadran"
    assert proba_oof.index.equals(y.index) and y.index.equals(fold_oof.index), "Índices temporales no alinean"

    proba_df = pd.DataFrame({
        "date": dates.to_series().reset_index(drop=True),
        "proba": pd.Series(proba_oof, dtype=float).reset_index(drop=True),
        "y_true": pd.Series(y, dtype=int).reset_index(drop=True),
        "fold": pd.Series(fold_oof, dtype="Int64").reset_index(drop=True),
    })

    prob_path = art / "probabilities_ml.csv"
    proba_df.to_csv(prob_path, index=False, encoding="utf-8")
    print("    OK (proba→probabilities_ml.csv con y_true + fold)  filas=", len(proba_df))

    # Backtest sobre toda la serie de precios
    proba_bt = proba_oof.reindex(df.index)
    backtest = backtest_long_only(df["Close"], proba_bt, cfg.proba_threshold, cfg.fees_bps, cfg.slippage_bps)

    # summary
    summary = {
        "config": asdict(cfg),
        "avg_metrics": {},
        "backtest": backtest
    }
    with open(art / "summary_ml.json", "w") as f:
        json.dump(summary, f, indent=2)

    if cfg.plot_bvb:
        save_bvb(df, art / "bvb")
    log_ok(t0, f"proba→{prob_path.name}")

    log_step(6, 6, "Listo.")
    return df, proba_bt


def run_report(cfg: Config) -> None:
    print(f"Config: {cfg}", flush=True)
    if _reporting is None:
        log_err("No puedo importar reporting.py. Asegúrate de tener project/reporting.py y __init__.py.")
        print("Tip: ejecuta desde el raíz del repo:  python project\\hybrid_trader.py --run report", flush=True)
        return

    summ_path = pathlib.Path("artifacts/summary_ml.json")
    proba_path = pathlib.Path("artifacts/probabilities_ml.csv")
    if not summ_path.exists() or not proba_path.exists():
        print("Faltan artefactos. Ejecuta primero --run ml", flush=True)
        return

    try:
        _ = _reporting.load_summary(str(summ_path))
        _ = _reporting.load_probabilities(str(proba_path))
        t0 = log_step(1, 1, "Generando reporte (equity, drawdown, ROC/PR, calibración, sweep)")
        _reporting.main()
        log_ok(t0, "reporte en artifacts/report/")
    except Exception as e:
        log_err(f"Fallo generando reporte: {e!r}")


def run_hybrid(cfg: Config) -> None:
    _, _ = run_ml(cfg)
    print("Nota: Modo HYBRID — RL opcional no ejecutado en esta versión.", flush=True)


# ====== CLI ======
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run", choices=["ml", "rl", "hybrid", "report"], default="hybrid")
    p.add_argument("--ticker", type=str, help="Símbolo (ej: AAPL)")
    p.add_argument("--start", type=str, help="YYYY-MM-DD")
    p.add_argument("--end", type=str, help="YYYY-MM-DD")
    p.add_argument("--threshold", type=float, help="Umbral proba para backtest")
    p.add_argument("--rl_steps", type=int)
    p.add_argument("--bvb_len", type=int)
    p.add_argument("--bvb_bars_back", type=int)
    p.add_argument("--bvb_tline", type=float)
    p.add_argument("--plot-bvb", action="store_true")
    p.add_argument("--data-source", choices=["auto", "av", "yahoo", "stooq"])
    p.add_argument("--av-key", type=str)
    p.add_argument("--no-adjust", action="store_true", help="usar Close sin ajustar (AV)")
    p.add_argument("--quiet", action="store_true", help="menos logs en entrenamiento")
    return p.parse_args()


def make_config(ns: argparse.Namespace) -> Config:
    cfg = Config()
    if ns.ticker: cfg.ticker = ns.ticker
    if ns.start: cfg.start_date = ns.start
    if ns.end: cfg.end_date = ns.end
    if ns.threshold is not None: cfg.proba_threshold = ns.threshold
    if ns.rl_steps is not None: cfg.rl_steps = ns.rl_steps
    if ns.bvb_len is not None: cfg.bvb_len = ns.bvb_len
    if ns.bvb_bars_back is not None: cfg.bvb_bars_back = ns.bvb_bars_back
    if ns.bvb_tline is not None: cfg.bvb_tline = ns.bvb_tline
    if ns.plot_bvb: cfg.plot_bvb = True
    if ns.data_source: cfg.data_source = ns.data_source
    if ns.av_key: cfg.av_key = ns.av_key
    if ns.no_adjust: cfg.use_adjusted_close = False
    if ns.quiet: cfg.verbose_train = False
    return cfg


def main():
    ns = parse_args()
    cfg = make_config(ns)
    print(f"Config: {cfg}", flush=True)

    try:
        if ns.run == "ml":
            run_ml(cfg)
        elif ns.run == "report":
            run_report(cfg)
        elif ns.run == "hybrid":
            run_hybrid(cfg)
        else:
            print("Modo RL no implementado en esta versión; ejecuta --run ml o --run hybrid.", flush=True)
    except Exception as e:
        log_err(str(e))
        raise


if __name__ == "__main__":
    main()


# path: tests/test_probabilities_csv.py
import os
import pathlib
import pandas as pd

ART = pathlib.Path("artifacts")
CSV = ART / "probabilities_ml.csv"

def test_csv_exists_and_columns():
    assert CSV.exists(), "Ejecuta primero el entrenamiento para generar artifacts/probabilities_ml.csv"
    df = pd.read_csv(CSV)
    expected = {"date", "proba", "y_true", "fold"}
    assert expected.issubset(df.columns), f"Faltan columnas: {expected - set(df.columns)}"
    assert len(df) > 0, "CSV vacío"

def test_types_ranges_and_order():
    df = pd.read_csv(CSV)
    # date parseable y ordenado
    dates = pd.to_datetime(df["date"], errors="raise")
    assert dates.is_monotonic_increasing, "Las fechas no están en orden ascendente"

    # proba en [0,1] y no NaN
    assert df["proba"].notna().all(), "NaN en proba"
    assert ((df["proba"] >= 0.0) & (df["proba"] <= 1.0)).all(), "proba fuera de [0,1]"

    # y_true ∈ {0,1}
    y = df["y_true"]
    assert y.notna().all(), "NaN en y_true"
    uniq = set(int(v) for v in y.unique())
    assert uniq.issubset({0, 1}), f"Valores inválidos en y_true: {uniq}"

    # fold entero positivo (al menos 1 fold)
    fold = pd.Series(df["fold"]).astype("Int64")
    assert fold.notna().all(), "NaN en fold"
    assert (fold >= 1).all(), "fold debe ser >=1"
    assert fold.nunique() >= 1, "fold único insuficiente"

def test_alignment_lengths():
    df = pd.read_csv(CSV)
    assert len(df["date"]) == len(df["proba"]) == len(df["y_true"]) == len(df["fold"]), "Longitudes no cuadran"
