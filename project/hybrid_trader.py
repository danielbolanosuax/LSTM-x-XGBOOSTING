# path: project/hybrid_trader.py
from __future__ import annotations

# --- Fix SSL/certificados y evitar curl_cffi en yfinance (por qué: rutas con acentos rompen curl en Windows) ---
import os as _os
try:
    import certifi as _certifi
    _os.environ.setdefault("YFINANCE_USE_CURL_CFFI", "0")
    _ca = _certifi.where()
    _os.environ.setdefault("SSL_CERT_FILE", _ca)
    _os.environ.setdefault("CURL_CA_BUNDLE", _ca)
except Exception:
    pass

# --- Preferir gymnasium; caer a gym si no está (por qué: SB3≥2.0 usa gymnasium) ---
try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:
    try:
        import gym  # type: ignore
        from gym import spaces  # type: ignore
    except Exception:
        gym = None
        spaces = None

import math
import json
import time
import argparse
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Iterable, Dict, List

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

# Keras compat
try:
    from project.nn_compat import Sequential, LSTM, Dense, Dropout, Adam, EarlyStopping
except Exception:
    from tensorflow.keras.models import Sequential  # type: ignore
    from tensorflow.keras.layers import LSTM, Dense, Dropout  # type: ignore
    from tensorflow.keras.optimizers import Adam  # type: ignore
    from tensorflow.keras.callbacks import EarlyStopping  # type: ignore

import xgboost as xgb

# Proveedores de datos
try:
    from alpha_vantage.timeseries import TimeSeries
except Exception:
    TimeSeries = None
try:
    import yfinance as yf
except Exception:
    yf = None

# Reporting
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")


# =======================
# Config
# =======================
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
    # Alpha Vantage
    alpha_vantage_key_env: str = "ALPHA_VANTAGE_KEY"
    av_key: Optional[str] = None
    data_source: str = "av"         # "av" | "yahoo"
    use_adjusted_close: bool = True
    # Bulls vs Bears
    bvb_len: int = 14
    bvb_bars_back: int = 120
    bvb_tline: float = 80.0
    plot_bvb: bool = False


def set_seeds(seed: int) -> None:
    _os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def choose_scaler(kind: str):
    return StandardScaler() if kind == "standard" else MinMaxScaler()


# =======================
# Datos + Features
# =======================
def fetch_ohlcv(cfg: Config) -> pd.DataFrame:
    """Preferir Alpha Vantage (Daily Adjusted); fallback yfinance."""
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        df.index = pd.to_datetime(df.index); df = df.sort_index()
        df = df[df.index >= pd.to_datetime(cfg.start_date)]
        if cfg.end_date:
            df = df[df.index <= pd.to_datetime(cfg.end_date)]
        req = ["Open","High","Low","Close","Volume"]
        if any(c not in df.columns for c in req):
            raise RuntimeError("Faltan columnas OHLCV")
        return df.dropna()

    key = (cfg.av_key or _os.getenv(cfg.alpha_vantage_key_env, "")).strip()
    if cfg.data_source.lower() == "av" and key and TimeSeries is not None:
        ts = TimeSeries(key=key, output_format="pandas")
        for attempt in range(3):
            try:
                if cfg.use_adjusted_close:
                    data, _ = ts.get_daily_adjusted(symbol=cfg.ticker, outputsize="full")
                    data = data.rename(columns={
                        "1. open":"Open","2. high":"High","3. low":"Low",
                        "4. close":"Close","5. adjusted close":"Adj Close","6. volume":"Volume"
                    })
                    if "Adj Close" in data.columns:
                        data["Close"] = data["Adj Close"]
                else:
                    data, _ = ts.get_daily(symbol=cfg.ticker, outputsize="full")
                    data = data.rename(columns={
                        "1. open":"Open","2. high":"High","3. low":"Low","4. close":"Close","5. volume":"Volume"
                    })
                df = data[["Open","High","Low","Close","Volume"]]
                return _clean(df)
            except Exception as e:
                if attempt < 2:
                    time.sleep(15)  # por qué: rate-limit free
                else:
                    print(f"[AlphaVantage] fallo: {e}")

    if yf is not None:
        try:
            df = yf.download(cfg.ticker, start=cfg.start_date, end=cfg.end_date, auto_adjust=True)
            df = df.rename(columns=str.title)[["Open","High","Low","Close","Volume"]]
            return _clean(df)
        except Exception as e:
            print(f"[yfinance] fallo: {e}")

    raise RuntimeError("No se pudieron obtener datos OHLCV con AV ni yfinance.")


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def rsi_ema(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    roll_up = pd.Series(up, index=close.index).ewm(span=n, adjust=False).mean()
    roll_dn = pd.Series(dn, index=close.index).ewm(span=n, adjust=False).mean()
    rs = roll_up / (roll_dn + 1e-12)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    fast = ema(close, 12); slow = ema(close, 26)
    line = fast - slow; sig = ema(line, 9); hist = line - sig
    return line, sig, hist


def stochastic_kd(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14):
    lo = l.rolling(n).min(); hi = h.rolling(n).max()
    k = 100 * (c - lo) / (hi - lo + 1e-12); d = k.rolling(3).mean()
    return k, d


def bulls_bears(close: pd.Series, high: pd.Series, low: pd.Series, length: int, bars_back: int):
    ma = ema(close, length)
    bulls = high - ma; bears = ma - low
    mb = bulls.rolling(bars_back).min(); xb = bulls.rolling(bars_back).max()
    nb = ((bulls - mb) / (xb - mb + 1e-12) - 0.5) * 100.0
    mb2 = bears.rolling(bars_back).min(); xb2 = bears.rolling(bars_back).max()
    nr = ((bears - mb2) / (xb2 - mb2 + 1e-12) - 0.5) * 100.0
    total = nb - nr
    return total, nb, nr


def institutional_index(volume: pd.Series, n: int = 50) -> pd.Series:
    return volume / (volume.rolling(n).mean() + 1e-12)


def add_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = df.copy()
    out["Return"] = out["Close"].pct_change()
    out["LogRet"] = np.log1p(out["Return"].fillna(0))
    out["RSI"] = rsi_ema(out["Close"], 14)
    out["MACD"], out["MACD_Signal"], out["MACD_Hist"] = macd(out["Close"])
    out["StochK"], out["StochD"] = stochastic_kd(out["High"], out["Low"], out["Close"], 14)
    out["Volatility20"] = out["LogRet"].rolling(20).std().fillna(0)
    out["InstIdx"] = institutional_index(out["Volume"], 50)
    # Medias móviles (tendencia)
    out["MA20"]  = out["Close"].rolling(20).mean()
    out["MA50"]  = out["Close"].rolling(50).mean()
    out["MA200"] = out["Close"].rolling(200).mean()
    # BvB
    t, nb, nr = bulls_bears(out["Close"], out["High"], out["Low"], cfg.bvb_len, cfg.bvb_bars_back)
    out["BvB_Total"], out["BvB_NormBulls"], out["BvB_NormBears"] = t, nb, nr
    out["BvB_Bullish"] = (out["BvB_Total"] > cfg.bvb_tline).astype(int)
    out["BvB_Bearish"] = (out["BvB_Total"] < -cfg.bvb_tline).astype(int)

    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    out.ffill(inplace=True); out.dropna(inplace=True)
    return out


# =======================
# Labels (triple-barrier)
# =======================
def triple_barrier_labels(close: pd.Series, horizon: int, k: float, vol: pd.Series) -> pd.Series:
    labels = np.zeros(len(close), dtype=int)
    c = close.values; sig = vol.fillna(vol.median()).values
    up_mult = (1 + k * sig); dn_mult = (1 - k * sig)
    for i in range(len(c)):
        t_end = min(i + horizon, len(c) - 1)
        up = c[i] * up_mult[i]; dn = c[i] * dn_mult[i]
        hit_up = False; hit_dn = False
        for j in range(i+1, t_end+1):
            if c[j] >= up: hit_up = True; break
            if c[j] <= dn: hit_dn = True; break
        labels[i] = 1 if (hit_up and not hit_dn) else 0
    labels[-horizon:] = 0
    return pd.Series(labels, index=close.index)


# =======================
# TimeSeries Split (purged)
# =======================
class PurgedWalkForwardSplit:
    def __init__(self, n_splits: int = 5, gap: int = 0):
        self.n_splits = n_splits; self.gap = gap
    def split(self, X: pd.DataFrame) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        n = len(X); fold = n // (self.n_splits + 1)
        for k in range(self.n_splits):
            train_end = fold * (k + 1)
            test_start = train_end + self.gap
            test_end = min(test_start + fold, n)
            if test_end <= test_start: continue
            yield np.arange(0, train_end), np.arange(test_start, test_end)


# =======================
# LSTM utils
# =======================
def make_sequences(X: np.ndarray, y: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    Xs, ys = [], []
    for i in range(window, len(X)):
        Xs.append(X[i-window:i, :]); ys.append(y[i])
    return np.asarray(Xs), np.asarray(ys)


def build_lstm(timesteps: int, features: int, lr: float = 1e-3) -> Sequential:
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(timesteps, features)),
        Dropout(0.2),  # por qué: reduce sobreajuste
        LSTM(64),
        Dropout(0.2),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss="binary_crossentropy", metrics=["accuracy"])
    return model


class LstmXgbStack:
    def __init__(self, cfg: Config):
        self.cfg = cfg; self.scaler = None; self.lstm = None; self.xgb = None
        self.feature_names_: List[str] = []

    def fit(self, Xtr_df: pd.DataFrame, ytr: np.ndarray, Xva_df: pd.DataFrame, yva: np.ndarray) -> None:
        self.feature_names_ = list(Xtr_df.columns)
        self.scaler = choose_scaler(self.cfg.scaler_type)
        Xtr = self.scaler.fit_transform(Xtr_df.values)
        Xva = self.scaler.transform(Xva_df.values)

        Xtr_seq, ytr_seq = make_sequences(Xtr, ytr, self.cfg.window)
        Xva_seq, yva_seq = make_sequences(Xva, yva, self.cfg.window)

        self.lstm = build_lstm(self.cfg.window, Xtr_seq.shape[2])
        es = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
        self.lstm.fit(Xtr_seq, ytr_seq, validation_data=(Xva_seq, yva_seq),
                      epochs=80, batch_size=64, callbacks=[es], verbose=0)

        p_tr = self.lstm.predict(Xtr_seq, verbose=0).ravel()
        p_va = self.lstm.predict(Xva_seq, verbose=0).ravel()

        Xtr_al = Xtr[self.cfg.window:, :]; Xva_al = Xva[self.cfg.window:, :]
        ytr_al = ytr[self.cfg.window:];   yva_al = yva[self.cfg.window:]

        Xtr_stack = np.c_[Xtr_al, p_tr]; Xva_stack = np.c_[Xva_al, p_va]

        pos = ytr_al.sum(); neg = len(ytr_al) - pos
        spw = (neg / max(1, pos)) if pos > 0 else 1.0

        self.xgb = xgb.XGBClassifier(
            n_estimators=1000, learning_rate=0.03, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            tree_method="hist", scale_pos_weight=spw, eval_metric="logloss",
            random_state=self.cfg.seed
        )
        self.xgb.fit(Xtr_stack, ytr_al, eval_set=[(Xva_stack, yva_al)],
                     early_stopping_rounds=100, verbose=False)

    def predict_proba_series(self, X_df: pd.DataFrame) -> np.ndarray:
        Xs = self.scaler.transform(X_df.values)
        X_seq, _ = make_sequences(Xs, np.zeros(len(Xs)), self.cfg.window)
        p_lstm = self.lstm.predict(X_seq, verbose=0).ravel()
        X_al = Xs[self.cfg.window:, :]
        X_stack = np.c_[X_al, p_lstm]
        p = self.xgb.predict_proba(X_stack)[:, 1]
        return np.r_[np.full(self.cfg.window, np.nan), p]


# =======================
# Métricas + Backtest
# =======================
def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    m = ~np.isnan(y_score)
    y_true = y_true[m]; y_score = y_score[m]
    y_pred = (y_score >= thr).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score)
    }


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
    dd = equity / equity.cummax() - 1.0
    cagr = equity.iloc[-1] ** (252/len(equity)) - 1.0 if len(equity) else 0.0
    sharpe = np.sqrt(252) * (net.mean() / (net.std() + 1e-12))
    return {"CAGR": float(cagr), "Sharpe": float(sharpe), "MaxDD": float(dd.min()),
            "HitRate": float((gross > 0).mean()), "TurnsPerYear": float(turns.sum() * (252/len(turns)))}


# =======================
# RL Env continuo (PPO)
# =======================
class TradingEnvCont(gym.Env if gym is not None else object):
    """Acción [-1,1] = posición objetivo; smoothing EMA; coste por turnover; recompensa en log."""
    metadata = {"render.modes": []}
    def __init__(self, X: np.ndarray, prices: np.ndarray, cfg: Config, p_alpha: Optional[np.ndarray] = None):
        if gym is None:
            raise RuntimeError("Gym/Gymnasium no disponible. `pip install gymnasium` o `gym`.")
        super().__init__()
        self.cfg = cfg; self.X = X.astype(np.float32); self.P = prices.astype(np.float32)
        self.W = cfg.window; self.t = self.W; self.equity = 1.0; self.pos = 0.0; self.smooth_pos = 0.0
        fdim = X.shape[1]; extra = 2
        self.p_alpha = p_alpha if p_alpha is not None else np.full(len(self.P), np.nan, dtype=np.float32)
        low = -np.inf * np.ones((self.W, fdim + extra), dtype=np.float32)
        high = np.inf * np.ones((self.W, fdim + extra), dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-1.0], dtype=np.float32), high=np.array([1.0], dtype=np.float32), dtype=np.float32)

    def _obs(self):
        s = self.t - self.W
        Xw = self.X[s:self.t, :]
        lr = np.log(self.P[1:] / self.P[:-1]); vol = pd.Series(lr).rolling(20).std().fillna(0).values
        vol_w = vol[s:self.t][:, None]; pa_w = np.nan_to_num(self.p_alpha[s:self.t], nan=0.0)[:, None]
        return np.concatenate([Xw, vol_w, pa_w], axis=1).astype(np.float32)

    def step(self, action: np.ndarray):
        a = float(np.clip(action[0], -1.0, 1.0))
        a = np.clip(a, self.smooth_pos - self.cfg.max_pos_change, self.smooth_pos + self.cfg.max_pos_change)
        self.smooth_pos = (1 - self.cfg.action_smooth_tau) * self.smooth_pos + self.cfg.action_smooth_tau * a
        ret = (self.P[self.t] / self.P[self.t-1]) - 1.0
        dpos = abs(self.smooth_pos - self.pos); self.pos = self.smooth_pos
        cost = dpos * ((self.cfg.fees_bps + self.cfg.slippage_bps) / 1e4)
        net = self.pos * ret - cost
        self.equity *= (1.0 + net)
        reward = math.log(1.0 + net) - self.cfg.rl_lambda_risk * (abs(self.pos) * abs(ret))
        self.t += 1; done = (self.t >= len(self.P) - 1)
        return self._obs(), reward, done, {"equity": self.equity, "position": self.pos, "cost": cost}

    def reset(self, seed: Optional[int] = None):
        self.t = self.W; self.equity = 1.0; self.pos = 0.0; self.smooth_pos = 0.0
        return self._obs()


# =======================
# Pipeline principal ML/RL
# =======================
FEATURES = [
    "Return","LogRet",
    "RSI","MACD","MACD_Signal","MACD_Hist",
    "StochK","StochD",
    "Volatility20","InstIdx",
    "MA20","MA50","MA200",
    "BvB_Total","BvB_NormBulls","BvB_NormBears",
]

def plot_bvb(df: pd.DataFrame, cfg: Config, outdir: str = "artifacts/bvb") -> None:
    _os.makedirs(outdir, exist_ok=True)
    cols = ["BvB_Total","BvB_Bullish","BvB_Bearish","BvB_NormBulls","BvB_NormBears"]
    df[cols].to_csv(_os.path.join(outdir, "bvb_series.csv"))
    plt.figure()
    df["BvB_Total"].plot(title=f"BvB_Total ({cfg.ticker})")
    bull_idx = df.index[df["BvB_Bullish"] == 1]
    bear_idx = df.index[df["BvB_Bearish"] == 1]
    plt.scatter(bull_idx, df.loc[bull_idx, "BvB_Total"])
    plt.scatter(bear_idx, df.loc[bear_idx, "BvB_Total"])
    plt.tight_layout(); plt.savefig(_os.path.join(outdir, "bvb_total.png")); plt.close()

def run_ml(cfg: Config) -> Tuple[pd.DataFrame, pd.Series]:
    set_seeds(cfg.seed)
    raw = fetch_ohlcv(cfg)
    df = add_features(raw, cfg)
    if cfg.plot_bvb: plot_bvb(df, cfg, outdir="artifacts/bvb")

    labels = triple_barrier_labels(df["Close"], cfg.horizon, cfg.barrier_mult, df["Volatility20"])
    df["y"] = labels

    X = df[FEATURES].copy(); y = df["y"].astype(int).values; idx = df.index
    splitter = PurgedWalkForwardSplit(n_splits=cfg.test_splits, gap=cfg.gap)
    fold_metrics, fold_preds = [], []
    for fi, (tr, te) in enumerate(splitter.split(X), start=1):
        Xtr, Xte = X.iloc[tr], X.iloc[te]; ytr, yte = y[tr], y[te]
        model = LstmXgbStack(cfg); model.fit(Xtr, ytr, Xte, yte)
        p_te = model.predict_proba_series(Xte); m = compute_metrics(yte, p_te, thr=0.5)
        fold_metrics.append(m)
        ser = pd.Series(np.nan, index=idx); ser.iloc[te] = p_te; fold_preds.append(ser)
        print(f"[Fold {fi}] " + ", ".join(f"{k}:{v:.4f}" for k,v in m.items()))

    avg = {k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0]}
    print("\n=== MÉTRICAS PROMEDIO (walk-forward) ===")
    for k, v in avg.items(): print(f"{k}: {v:.4f}")

    all_proba = pd.concat(fold_preds, axis=1).bfill(axis=1).iloc[:,0]
    bt = backtest_long_only(df["Close"], all_proba, cfg.proba_threshold, cfg.fees_bps, cfg.slippage_bps)
    print("\n=== BACKTEST ML (long-only) ===")
    for k, v in bt.items(): print(f"{k}: {v:.4f}")

    _os.makedirs("artifacts", exist_ok=True)
    all_proba.to_csv("artifacts/probabilities_ml.csv")
    with open("artifacts/summary_ml.json","w") as f:
        json.dump({"config": cfg.__dict__, "avg_metrics": avg, "backtest": bt}, f, indent=2)
    print("Guardado: artifacts/probabilities_ml.csv, artifacts/summary_ml.json")
    return df, all_proba


def run_rl(cfg: Config, df: pd.DataFrame, proba: Optional[pd.Series] = None) -> None:
    if gym is None:
        raise RuntimeError("Falta gymnasium/gym o stable-baselines3.")
    scaler = choose_scaler(cfg.scaler_type)
    X = scaler.fit_transform(df[FEATURES].values)
    P = df["Close"].values.astype(np.float32)
    p_alpha = None if proba is None else proba.values.astype(np.float32)
    env = TradingEnvCont(X, P, cfg, p_alpha=p_alpha)
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        venv = DummyVecEnv([lambda: env])
        model = PPO("MlpPolicy", venv, verbose=1, seed=cfg.seed, n_steps=2048, batch_size=256)
        model.learn(total_timesteps=cfg.rl_steps)
        _os.makedirs("artifacts", exist_ok=True)
        model.save("artifacts/ppo_hybrid.zip")
        print("Modelo PPO guardado en artifacts/ppo_hybrid.zip")
    except Exception as e:
        print(f"No pude entrenar PPO (instala stable-baselines3): {e}")


def run_hybrid(cfg: Config) -> None:
    df, proba = run_ml(cfg)
    run_rl(cfg, df, proba)


# =======================
# ======= REPORTING =====
# =======================
def compute_drawdown(equity: pd.Series) -> pd.Series:
    peak = equity.cummax(); return equity/peak - 1.0

def plot_equity_and_drawdown(close: pd.Series, proba: pd.Series, thr: float, fees_bps: float, slippage_bps: float, outdir: str) -> Dict[str, float]:
    bt = backtest_long_only(close, proba, thr, fees_bps, slippage_bps)
    proba = proba.fillna(0.0); signal = (proba >= thr).astype(int)
    pos = signal.shift(1).fillna(0); ret = close.pct_change().fillna(0.0)
    net = (pos * ret) - pos.diff().abs().fillna(0.0) * ((fees_bps + slippage_bps)/1e4)
    equity = (1 + net).cumprod(); dd = compute_drawdown(equity)
    _os.makedirs(outdir, exist_ok=True)
    plt.figure(); equity.plot(title="Equity curve (long-only, ML)"); plt.xlabel("Fecha"); plt.ylabel("Equity")
    plt.tight_layout(); plt.savefig(_os.path.join(outdir, "equity_curve.png")); plt.close()
    plt.figure(); dd.plot(title="Drawdown"); plt.xlabel("Fecha"); plt.ylabel("Drawdown")
    plt.tight_layout(); plt.savefig(_os.path.join(outdir, "drawdown.png")); plt.close()
    return bt

def plot_roc_pr(y_true: pd.Series, y_score: pd.Series, outdir: str) -> Dict[str, float]:
    mask = (~y_score.isna()); y = y_true[mask].astype(int).values; s = y_score[mask].astype(float).values
    fpr, tpr, _ = roc_curve(y, s); roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y, s); ap = average_precision_score(y, s)
    plt.figure(); plt.plot(fpr, tpr); plt.plot([0,1], [0,1], linestyle="--")
    plt.title(f"ROC (AUC={roc_auc:.3f})"); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.tight_layout(); plt.savefig(_os.path.join(outdir, "roc.png")); plt.close()
    plt.figure(); plt.plot(recall, precision)
    plt.title(f"Precision-Recall (AP={ap:.3f})"); plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.tight_layout(); plt.savefig(_os.path.join(outdir, "pr.png")); plt.close()
    thr = 0.5; yhat = (s >= thr).astype(int)
    cm = confusion_matrix(y, yhat); rep = classification_report(y, yhat, zero_division=0, output_dict=True)
    pd.DataFrame(cm, index=["True 0","True 1"], columns=["Pred 0","Pred 1"]).to_csv(_os.path.join(outdir, "confusion_0.5.csv"))
    pd.DataFrame(rep).to_csv(_os.path.join(outdir, "class_report_0.5.csv"))
    return {"roc_auc": float(roc_auc), "pr_ap": float(ap)}

def plot_calibration(y_true: pd.Series, y_score: pd.Series, outdir: str, n_bins: int = 10) -> None:
    mask = (~y_score.isna()); y = y_true[mask].astype(int).values; s = y_score[mask].astype(float).values
    bins = np.linspace(0.0, 1.0, n_bins + 1); idx = np.digitize(s, bins) - 1
    df = pd.DataFrame({"y": y, "s": s, "bin": idx})
    grp = df.groupby("bin").agg(emp_proba=("y","mean"), mean_pred=("s","mean"), count=("y","size"))
    grp = grp[(grp.index >= 0) & (grp.index < n_bins)]
    plt.figure(); plt.plot([0,1], [0,1], linestyle="--"); plt.plot(grp["mean_pred"], grp["emp_proba"])
    plt.title("Calibración (pred vs empírica)"); plt.xlabel("Probabilidad predicha"); plt.ylabel("Tasa empírica")
    plt.tight_layout(); plt.savefig(_os.path.join(outdir, "calibration.png")); plt.close()
    grp.to_csv(_os.path.join(outdir, "calibration_bins.csv"))

def threshold_sweep(close: pd.Series, proba: pd.Series, fees_bps: float, slippage_bps: float, outdir: str) -> pd.DataFrame:
    thrs = np.linspace(0.4, 0.8, 21); rows = []
    for t in thrs:
        bt = backtest_long_only(close, proba, float(t), fees_bps, slippage_bps); rows.append({"threshold": float(t), **bt})
    df = pd.DataFrame(rows)
    plt.figure(); plt.plot(df["threshold"], df["CAGR"]); plt.title("CAGR vs threshold"); plt.xlabel("threshold"); plt.ylabel("CAGR")
    plt.tight_layout(); plt.savefig(_os.path.join(outdir, "sweep_cagr.png")); plt.close()
    plt.figure(); plt.plot(df["threshold"], df["Sharpe"]); plt.title("Sharpe vs threshold"); plt.xlabel("threshold"); plt.ylabel("Sharpe")
    plt.tight_layout(); plt.savefig(_os.path.join(outdir, "sweep_sharpe.png")); plt.close()
    df.to_csv(_os.path.join(outdir, "threshold_sweep.csv"), index=False)
    return df

def generate_report(close: pd.Series, vol20: pd.Series, proba: pd.Series,
                    horizon: int, barrier_mult: float, threshold: float,
                    fees_bps: float, slippage_bps: float,
                    outdir: str = "artifacts/report",
                    df_bvb: Optional[pd.DataFrame] = None, cfg: Optional[Config] = None) -> None:
    _os.makedirs(outdir, exist_ok=True)
    y = triple_barrier_labels(close, horizon=horizon, k=barrier_mult, vol=vol20)
    bt = plot_equity_and_drawdown(close, proba, threshold, fees_bps, slippage_bps, outdir)
    curves = plot_roc_pr(y, proba, outdir)
    plot_calibration(y, proba, outdir)
    sweep = threshold_sweep(close, proba, fees_bps, slippage_bps, outdir)
    if df_bvb is not None and cfg is not None:
        plot_bvb(df_bvb, cfg, outdir=_os.path.join(outdir, "bvb"))
    with open(_os.path.join(outdir, "report_summary.json"), "w") as f:
        json.dump({
            "classification": curves,
            "backtest@threshold": {**bt, "threshold": threshold},
            "best_threshold_by_cagr": float(sweep.iloc[sweep["CAGR"].idxmax()]["threshold"]),
            "best_threshold_by_sharpe": float(sweep.iloc[sweep["Sharpe"].idxmax()]["threshold"])
        }, f, indent=2)
    with open(_os.path.join(outdir, "README.txt"), "w") as f:
        f.write(
            "Artículos generados:\n"
            "- equity_curve.png, drawdown.png\n"
            "- roc.png, pr.png, confusion_0.5.csv, class_report_0.5.csv\n"
            "- calibration.png, calibration_bins.csv\n"
            "- sweep_cagr.png, sweep_sharpe.png, threshold_sweep.csv\n"
            "- bvb/bvb_total.png, bvb/bvb_series.csv\n"
            "- report_summary.json\n"
        )

def run_report() -> None:
    if not _os.path.exists("artifacts/probabilities_ml.csv") or not _os.path.exists("artifacts/summary_ml.json"):
        print("Faltan artefactos. Ejecuta primero --run ml"); return
    with open("artifacts/summary_ml.json","r") as f:
        summ = json.load(f)
    cfg = Config(**summ["config"])

    proba = pd.read_csv("artifacts/probabilities_ml.csv", index_col=0).iloc[:,0]
    proba.index = pd.to_datetime(proba.index); proba.name = "proba"

    df = fetch_ohlcv(cfg)  # misma fuente que training
    # reconstruir Vol20 para labels del reporte
    tmp = df[["Open","High","Low","Close","Volume"]].copy()
    tmp["LogRet"] = np.log(tmp["Close"]).diff().fillna(0)
    tmp["Volatility20"] = tmp["LogRet"].rolling(20).std().fillna(0)
    tmp = tmp.reindex(proba.index).dropna()
    proba = proba.reindex(tmp.index)
    df_bvb = add_features(df, cfg)[["BvB_Total","BvB_Bullish","BvB_Bearish","BvB_NormBulls","BvB_NormBears"]].reindex(tmp.index)

    generate_report(
        close=tmp["Close"], vol20=tmp["Volatility20"], proba=proba,
        horizon=int(cfg.horizon), barrier_mult=float(cfg.barrier_mult),
        threshold=float(cfg.proba_threshold), fees_bps=float(cfg.fees_bps), slippage_bps=float(cfg.slippage_bps),
        outdir="artifacts/report", df_bvb=df_bvb, cfg=cfg
    )
    print("Reporte generado en artifacts/report/")


# =======================
# CLI
# =======================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run", choices=["ml","rl","hybrid","report"], default="hybrid")
    p.add_argument("--ticker", type=str); p.add_argument("--start", type=str); p.add_argument("--end", type=str)
    p.add_argument("--threshold", type=float); p.add_argument("--rl_steps", type=int)
    # BvB
    p.add_argument("--bvb_len", type=int); p.add_argument("--bvb_bars_back", type=int); p.add_argument("--bvb_tline", type=float)
    p.add_argument("--plot-bvb", action="store_true")
    # Fuente de datos
    p.add_argument("--data-source", choices=["av","yahoo"])
    p.add_argument("--av-key", type=str)
    p.add_argument("--no-adjust", action="store_true", help="usar Close sin ajustar (AV)")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = Config()
    if args.ticker: cfg.ticker = args.ticker
    if args.start: cfg.start_date = args.start
    if args.end: cfg.end_date = args.end
    if args.threshold is not None: cfg.proba_threshold = args.threshold
    if args.rl_steps: cfg.rl_steps = args.rl_steps
    if args.bvb_len: cfg.bvb_len = args.bvb_len
    if args.bvb_bars_back: cfg.bvb_bars_back = args.bvb_bars_back
    if args.bvb_tline: cfg.bvb_tline = args.bvb_tline
    if args.plot_bvb: cfg.plot_bvb = True
    if args.data_source: cfg.data_source = args.data_source
    if args.av_key: cfg.av_key = args.av_key
    if args.no_adjust: cfg.use_adjusted_close = False

    print(f"Config: {cfg}")
    if args.run == "ml":
        run_ml(cfg)
    elif args.run == "rl":
        df = add_features(fetch_ohlcv(cfg), cfg)
        if cfg.plot_bvb: plot_bvb(df, cfg, outdir="artifacts/bvb")
        run_rl(cfg, df, None)
    elif args.run == "report":
        run_report()
    else:
        df, proba = run_ml(cfg)
        run_rl(cfg, df, proba)

if __name__ == "__main__":
    main()
