# path: hybrid_trader.py
from __future__ import annotations

import os
import math
import json
import argparse
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Iterable, Dict, List

import numpy as np
import pandas as pd

# ML
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)
from project.nn_compat import Sequential, LSTM, Dense, Dropout, Adam, EarlyStopping

import xgboost as xgb

# Data
try:
    import yfinance as yf
except Exception:
    yf = None
try:
    from alpha_vantage.timeseries import TimeSeries
except Exception:
    TimeSeries = None

# RL
try:
    import gym
    from gym import spaces
except Exception:
    gym = None

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
    scaler_type: str = "standard"  # standard|minmax
    test_splits: int = 5
    gap: int = 3
    # Costs
    fees_bps: float = 5.0
    slippage_bps: float = 5.0
    # ML
    proba_threshold: float = 0.6
    seed: int = 42
    # RL
    rl_lambda_risk: float = 0.0
    rl_steps: int = 200_000
    action_smooth_tau: float = 0.2    # por qué: reduce turnover
    max_pos_change: float = 0.3       # por qué: limita saltos de posición
    vol_target: float = 0.2           # por qué: target anualizado aprox
    # IO
    alpha_vantage_key_env: str = "ALPHA_VANTAGE_KEY"


def set_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def choose_scaler(kind: str):
    return StandardScaler() if kind == "standard" else MinMaxScaler()


# =======================
# Datos + Features
# =======================
def fetch_ohlcv(cfg: Config) -> pd.DataFrame:
    key = os.getenv(cfg.alpha_vantage_key_env, "").strip()
    df = None
    if key and TimeSeries is not None:
        try:
            ts = TimeSeries(key=key, output_format="pandas")
            data, _ = ts.get_daily(symbol=cfg.ticker, outputsize="full")
            data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            df = data.sort_index()
        except Exception:
            df = None
    if df is None and yf is not None:
        df = yf.download(cfg.ticker, start=cfg.start_date, end=cfg.end_date)
        df = df[["Open","High","Low","Close","Volume"]]
    if df is None or df.empty:
        raise RuntimeError("No se pudieron obtener datos OHLCV.")
    df.index = pd.to_datetime(df.index)
    df = df[df.index >= pd.to_datetime(cfg.start_date)]
    if cfg.end_date:
        df = df[df.index <= pd.to_datetime(cfg.end_date)]
    return df.dropna()


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def rsi_ema(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=close.index).ewm(span=n, adjust=False).mean()
    roll_down = pd.Series(down, index=close.index).ewm(span=n, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    fast = ema(close, 12)
    slow = ema(close, 26)
    macd_line = fast - slow
    signal = ema(macd_line, 9)
    hist = macd_line - signal
    return macd_line, signal, hist


def stochastic_kd(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14):
    lowest = low.rolling(n).min()
    highest = high.rolling(n).max()
    k = 100 * (close - lowest) / (highest - lowest + 1e-12)
    d = k.rolling(3).mean()
    return k, d


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    pc = close.shift(1)
    tr = pd.concat([(high - low), (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume).cumsum()


def institutional_index(volume: pd.Series, n: int = 50) -> pd.Series:
    ma = volume.rolling(n).mean()
    return volume / (ma + 1e-12)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Return"] = out["Close"].pct_change()
    out["LogRet"] = np.log1p(out["Return"].fillna(0))
    out["RSI"] = rsi_ema(out["Close"], 14)
    out["MACD"], out["MACD_Signal"], out["MACD_Hist"] = macd(out["Close"])
    out["StochK"], out["StochD"] = stochastic_kd(out["High"], out["Low"], out["Close"], 14)
    out["ATR14"] = atr(out["High"], out["Low"], out["Close"], 14)
    out["Volatility20"] = out["LogRet"].rolling(20).std().fillna(0)
    out["OBV"] = obv(out["Close"], out["Volume"])
    out["MA20"] = out["Close"].rolling(20).mean()
    out["MA50"] = out["Close"].rolling(50).mean()
    out["MA200"] = out["Close"].rolling(200).mean()
    out["InstIdx"] = institutional_index(out["Volume"], 50)
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    out.ffill(inplace=True)
    out.dropna(inplace=True)
    return out


# =======================
# Labels (triple-barrier)
# =======================
def triple_barrier_labels(close: pd.Series, horizon: int, k: float, vol: pd.Series) -> pd.Series:
    labels = np.zeros(len(close), dtype=int)
    c = close.values
    sig = vol.fillna(vol.median()).values
    up_mult = (1 + k * sig)
    dn_mult = (1 - k * sig)
    for i in range(len(c)):
        t_end = min(i + horizon, len(c) - 1)
        up = c[i] * up_mult[i]
        dn = c[i] * dn_mult[i]
        hit_up = False
        hit_dn = False
        for j in range(i+1, t_end+1):
            if c[j] >= up:
                hit_up = True
                break
            if c[j] <= dn:
                hit_dn = True
                break
        labels[i] = 1 if (hit_up and not hit_dn) else 0
    labels[-horizon:] = 0
    return pd.Series(labels, index=close.index)


# =======================
# TimeSeries Split (purged)
# =======================
class PurgedWalkForwardSplit:
    def __init__(self, n_splits: int = 5, gap: int = 0):
        self.n_splits = n_splits
        self.gap = gap

    def split(self, X: pd.DataFrame) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        n = len(X)
        fold = n // (self.n_splits + 1)
        for k in range(self.n_splits):
            train_end = fold * (k + 1)
            test_start = train_end + self.gap
            test_end = min(test_start + fold, n)
            if test_end <= test_start:
                continue
            yield np.arange(0, train_end), np.arange(test_start, test_end)


# =======================
# LSTM utils
# =======================
def make_sequences(X: np.ndarray, y: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    Xs, ys = [], []
    for i in range(window, len(X)):
        Xs.append(X[i-window:i, :])
        ys.append(y[i])
    return np.asarray(Xs), np.asarray(ys)


def build_lstm(input_timesteps: int, input_features: int, lr: float = 1e-3) -> Sequential:
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(input_timesteps, input_features)),
        Dropout(0.2),  # por qué: reduce sobreajuste
        LSTM(64),
        Dropout(0.2),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss="binary_crossentropy", metrics=["accuracy"])
    return model


class LstmXgbStack:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.scaler = None
        self.lstm = None
        self.xgb = None
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

        Xtr_al = Xtr[self.cfg.window:, :]
        Xva_al = Xva[self.cfg.window:, :]
        ytr_al = ytr[self.cfg.window:]
        yva_al = yva[self.cfg.window:]

        Xtr_stack = np.c_[Xtr_al, p_tr]
        Xva_stack = np.c_[Xva_al, p_va]

        pos = ytr_al.sum()
        neg = len(ytr_al) - pos
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
        pad = np.full(self.cfg.window, np.nan)
        return np.r_[pad, p]


# =======================
# Métricas + Backtest
# =======================
def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    mask = ~np.isnan(y_score)
    y_true = y_true[mask]
    y_score = y_score[mask]
    y_pred = (y_score >= thr).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score)
    }


def backtest_long_only(close: pd.Series, proba: pd.Series,
                       threshold: float, fees_bps: float, slippage_bps: float) -> Dict[str, float]:
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
    """
    Acción continua en [-1,1] = posición objetivo. Smoothing EMA y límite Δpos.
    Recompensa en log para estabilidad; coste por turnover.
    """
    metadata = {"render.modes": []}

    def __init__(self, X: np.ndarray, prices: np.ndarray, cfg: Config,
                 p_alpha: Optional[np.ndarray] = None):
        if gym is None:
            raise RuntimeError("Gym no disponible. `pip install gym`.")
        super().__init__()
        self.cfg = cfg
        self.X = X.astype(np.float32)
        self.P = prices.astype(np.float32)
        self.W = cfg.window
        self.t = self.W
        self.equity = 1.0
        self.pos = 0.0
        self.smooth_pos = 0.0  # por qué: reduce turnover

        fdim = X.shape[1]
        extra_dim = 2  # vol realiz + última proba alpha
        self.p_alpha = p_alpha if p_alpha is not None else np.full(len(self.P), np.nan, dtype=np.float32)

        low = -np.inf * np.ones((self.W, fdim + extra_dim), dtype=np.float32)
        high = np.inf * np.ones((self.W, fdim + extra_dim), dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-1.0], dtype=np.float32),
                                       high=np.array([1.0], dtype=np.float32), dtype=np.float32)

    def _obs(self):
        # vol realizada local y alpha como canales adicionales
        start = self.t - self.W
        Xw = self.X[start:self.t, :]
        lr = np.log(self.P[1:] / self.P[:-1])
        vol = pd.Series(lr).rolling(20).std().fillna(0).values  # proxy
        vol_w = vol[start:self.t][:, None]
        pa_w = np.nan_to_num(self.p_alpha[start:self.t], nan=0.0)[:, None]
        return np.concatenate([Xw, vol_w, pa_w], axis=1).astype(np.float32)

    def step(self, action: np.ndarray):
        a = float(np.clip(action[0], -1.0, 1.0))
        # limitador de salto
        a = np.clip(a, self.smooth_pos - self.cfg.max_pos_change, self.smooth_pos + self.cfg.max_pos_change)
        # smoothing
        self.smooth_pos = (1 - self.cfg.action_smooth_tau) * self.smooth_pos + self.cfg.action_smooth_tau * a

        # volatilidad objetivo (scaling naive)
        ret = (self.P[self.t] / self.P[self.t-1]) - 1.0
        step_pos = self.smooth_pos
        gross = step_pos * ret

        # coste por turnover
        dpos = abs(step_pos - self.pos)
        self.pos = step_pos
        cost = dpos * ((self.cfg.fees_bps + self.cfg.slippage_bps) / 1e4)

        net = gross - cost
        self.equity *= (1.0 + net)
        reward = math.log(1.0 + net)  # robusto con retornos pequeños

        # penalización simple por riesgo
        reward -= self.cfg.rl_lambda_risk * (abs(step_pos) * abs(ret))

        self.t += 1
        done = (self.t >= len(self.P) - 1)
        info = {"equity": self.equity, "position": self.pos, "cost": cost}
        return self._obs(), reward, done, info

    def reset(self, seed: Optional[int] = None):
        self.t = self.W
        self.equity = 1.0
        self.pos = 0.0
        self.smooth_pos = 0.0
        return self._obs()


# =======================
# Pipeline principal
# =======================
def run_ml(cfg: Config) -> Tuple[pd.DataFrame, pd.Series]:
    set_seeds(cfg.seed)
    raw = fetch_ohlcv(cfg)
    df = add_features(raw)
    labels = triple_barrier_labels(df["Close"], horizon=cfg.horizon, k=cfg.barrier_mult, vol=df["Volatility20"])
    df["y"] = labels

    feats = ["RSI","MACD","MACD_Signal","MACD_Hist","StochK","StochD","ATR14","Volatility20",
             "OBV","MA20","MA50","MA200","InstIdx","Return"]
    X = df[feats].copy()
    y = df["y"].astype(int).values
    idx = df.index

    splitter = PurgedWalkForwardSplit(n_splits=cfg.test_splits, gap=cfg.gap)
    fold_metrics, fold_preds = [], []
    for fi, (tr, te) in enumerate(splitter.split(X), start=1):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y[tr], y[te]
        model = LstmXgbStack(cfg)
        model.fit(Xtr, ytr, Xte, yte)
        proba_te = model.predict_proba_series(Xte)
        m = compute_metrics(yte, proba_te, thr=0.5)
        fold_metrics.append(m)
        ser = pd.Series(np.nan, index=idx)
        ser.iloc[te] = proba_te
        fold_preds.append(ser)
        print(f"[Fold {fi}] " + ", ".join(f"{k}:{v:.4f}" for k,v in m.items()))

    avg = {k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0]}
    print("\n=== MÉTRICAS PROMEDIO (walk-forward) ===")
    for k, v in avg.items():
        print(f"{k}: {v:.4f}")

    all_proba = pd.concat(fold_preds, axis=1).bfill(axis=1).iloc[:,0]
    bt = backtest_long_only(df["Close"], all_proba, cfg.proba_threshold, cfg.fees_bps, cfg.slippage_bps)
    print("\n=== BACKTEST ML (long-only) ===")
    for k, v in bt.items():
        print(f"{k}: {v:.4f}")

    os.makedirs("artifacts", exist_ok=True)
    all_proba.to_csv("artifacts/probabilities_ml.csv")
    with open("artifacts/summary_ml.json","w") as f:
        json.dump({"config": cfg.__dict__, "avg_metrics": avg, "backtest": bt}, f, indent=2)
    print("Guardado: artifacts/probabilities_ml.csv, artifacts/summary_ml.json")

    return df, all_proba


def run_rl(cfg: Config, df: pd.DataFrame, proba: Optional[pd.Series] = None) -> None:
    if gym is None:
        raise RuntimeError("Falta gym / stable-baselines3 para RL.")
    feats = ["RSI","MACD","MACD_Signal","MACD_Hist","StochK","StochD","ATR14","Volatility20",
             "OBV","MA20","MA50","MA200","InstIdx","Return"]
    scaler = choose_scaler(cfg.scaler_type)
    X = scaler.fit_transform(df[feats].values)
    P = df["Close"].values.astype(np.float32)
    p_alpha = None if proba is None else proba.values.astype(np.float32)

    env = TradingEnvCont(X, P, cfg, p_alpha=p_alpha)
    # Entrenamiento PPO (si está SB3)
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        venv = DummyVecEnv([lambda: env])
        model = PPO("MlpPolicy", venv, verbose=1, seed=cfg.seed, n_steps=2048, batch_size=256)
        model.learn(total_timesteps=cfg.rl_steps)
        os.makedirs("artifacts", exist_ok=True)
        model.save("artifacts/ppo_hybrid.zip")
        print("Modelo PPO guardado en artifacts/ppo_hybrid.zip")
    except Exception as e:
        print(f"No pude entrenar PPO (instala stable-baselines3): {e}")


def run_hybrid(cfg: Config) -> None:
    df, proba = run_ml(cfg)
    run_rl(cfg, df, proba)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run", choices=["ml","rl","hybrid"], default="hybrid")
    p.add_argument("--ticker", type=str)
    p.add_argument("--start", type=str)
    p.add_argument("--end", type=str)
    p.add_argument("--threshold", type=float)
    p.add_argument("--rl_steps", type=int)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config()
    if args.ticker: cfg.ticker = args.ticker
    if args.start: cfg.start_date = args.start
    if args.end: cfg.end_date = args.end
    if args.threshold is not None: cfg.proba_threshold = args.threshold
    if args.rl_steps: cfg.rl_steps = args.rl_steps

    print(f"Config: {cfg}")
    if args.run == "ml":
        run_ml(cfg)
    elif args.run == "rl":
        df = add_features(fetch_ohlcv(cfg))
        run_rl(cfg, df, None)
    else:
        run_hybrid(cfg)


if __name__ == "__main__":
    main()
