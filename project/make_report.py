# project/make_report.py
# -*- coding: utf-8 -*-

import os, json, webbrowser, math
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss, log_loss,
    precision_recall_curve, roc_curve, confusion_matrix
)
from sklearn.calibration import calibration_curve

# ------------------------ paths ------------------------
ARTIFACTS_DIR = os.path.join("artifacts")
REPORT_DIR    = os.path.join(ARTIFACTS_DIR, "report")
IMG_DIR       = os.path.join(REPORT_DIR, "img")
PROBA_CSV     = os.path.join(ARTIFACTS_DIR, "probabilities_ml.csv")
SUMMARY_JSON  = os.path.join(ARTIFACTS_DIR, "summary_ml.json")
HTML_OUT      = os.path.join(REPORT_DIR, "report.html")

plt.rcParams.update({
    "figure.figsize": (9, 6),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

# ------------------------ utils ------------------------
def _ensure_dirs():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)

def _read_summary():
    cfg, meta = {}, {}
    if os.path.isfile(SUMMARY_JSON):
        with open(SUMMARY_JSON, "r", encoding="utf-8") as f:
            meta = json.load(f)
        cfg = meta.get("config", {}) or {}
    return cfg, meta

def smart_read_probabilities():
    """
    Lee artifacts/probabilities_ml.csv y detecta:
      date, proba, y_true (opcional: y / target / label)
    """
    if not os.path.isfile(PROBA_CSV):
        raise FileNotFoundError(f"No existe {PROBA_CSV}. Ejecuta el entrenamiento primero.")

    df = pd.read_csv(PROBA_CSV)

    # fecha
    date_col = None
    for c in ["date", "Date", "ds", "timestamp", "Datetime", "datetime"]:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        raise ValueError("No encuentro columna de fecha (date/Date/ds/timestamp) en probabilities_ml.csv")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    df = df.rename(columns={date_col: "date"})

    # proba
    proba_col = None
    for c in ["proba", "prob", "pred_proba", "y_pred_proba", "p1", "score"]:
        if c in df.columns:
            proba_col = c
            break
    if proba_col is None:
        raise ValueError("No encuentro columna de probabilidad (proba/prob/...).")
    df = df.rename(columns={proba_col: "proba"})

    # y_true (opcional)
    y_col = None
    for c in ["y_true", "y", "target", "label"]:
        if c in df.columns:
            y_col = c
            break
    if y_col:
        df = df.rename(columns={y_col: "y_true"})
        keep = ["date", "proba", "y_true"]
    else:
        keep = ["date", "proba"]

    out = df[keep].copy()
    meta = {"label_col_found": ("y_true" if y_col else None), "source_file": PROBA_CSV}
    return out, meta

def _compose_cache_filename(cfg: dict) -> str:
    cache_dir = cfg.get("cache_dir", "data")
    ticker = cfg.get("ticker", "TICK")
    start = (cfg.get("start_date") or "").replace("-", "")
    end   = (cfg.get("end_date") or "None")
    adj   = "_adj" if cfg.get("use_adjusted_close", True) else ""
    return os.path.join(cache_dir, f"{ticker}_{start}_{end}{adj}.csv")

def smart_read_prices(cfg: dict):
    """
    Devuelve DF con columnas: date, price
    Busca primero el cache típico en data/, y si no, cualquier <ticker>_*.csv en data/
    """
    paths = []
    candidate = _compose_cache_filename(cfg)
    if os.path.isfile(candidate):
        paths.append(candidate)

    data_dir = cfg.get("cache_dir", "data")
    if os.path.isdir(data_dir):
        for fn in os.listdir(data_dir):
            if fn.lower().startswith((cfg.get("ticker", "") or "").lower() + "_") and fn.lower().endswith(".csv"):
                p = os.path.join(data_dir, fn)
                if p not in paths:
                    paths.append(p)

    for p in paths:
        try:
            df = pd.read_csv(p)
            dcol = None
            for c in ["date", "Date", "timestamp", "Datetime", "datetime"]:
                if c in df.columns:
                    dcol = c
                    break
            if dcol is None:
                continue
            df[dcol] = pd.to_datetime(df[dcol])

            pcol = None
            for c in ["Adj Close", "adj_close", "AdjClose", "Close", "close", "Close*"]:
                if c in df.columns:
                    pcol = c
                    break
            if pcol is None:
                continue

            out = df[[dcol, pcol]].copy().rename(columns={dcol: "date", pcol: "price"})
            out = out.sort_values("date").reset_index(drop=True)
            return out
        except Exception:
            continue
    return None


def clean_probs(df: pd.DataFrame):
    """
    Limpia el DataFrame de probabilidades:
      - fuerza tipos numéricos
      - elimina filas con NaN en columnas críticas
      - clip proba a [0,1]
      - fuerza y_true a {0,1} si existe
      - quita duplicados por fecha
      - reordena por fecha
    """
    out = df.copy()

    if "date" not in out.columns:
        raise ValueError("clean_probs: falta columna 'date'")
    if "proba" not in out.columns:
        raise ValueError("clean_probs: falta columna 'proba'")

    out["proba"] = pd.to_numeric(out["proba"], errors="coerce")
    if "y_true" in out.columns:
        out["y_true"] = pd.to_numeric(out["y_true"], errors="coerce")

    subset_cols = ["proba"] + (["y_true"] if "y_true" in out.columns else [])
    n0 = len(out)
    out = out.dropna(subset=subset_cols)

    out["proba"] = out["proba"].clip(0.0, 1.0)

    if "y_true" in out.columns:
        out["y_true"] = out["y_true"].round().astype(int)
        out = out[out["y_true"].isin([0, 1])]

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"])
    out = out.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)

    return out, {"dropped": n0 - len(out)}

# ------------------------ métricas & plots ------------------------
def best_f1_threshold(y_true: np.ndarray, proba: np.ndarray):
    # Si no hay suficientes clases o datos, devolvemos algo neutro
    if y_true is None or proba is None:
        return 0.5, 0.0
    y_true = np.asarray(y_true)
    proba  = np.asarray(proba)

    # Necesitamos al menos una clase positiva y una negativa
    if len(np.unique(y_true)) < 2 or len(y_true) < 3:
        return 0.5, 0.0

    pr, rc, thr = precision_recall_curve(y_true, proba)
    # construir F1 para cada punto de umbral (la API devuelve N-1 umbrales)
    eps = 1e-12
    if len(pr) < 2 or len(rc) < 2 or len(thr) == 0:
        return 0.5, 0.0
    f1s = 2 * (pr[:-1] * rc[:-1]) / (pr[:-1] + rc[:-1] + eps)
    if len(f1s) == 0 or not np.isfinite(f1s).any():
        return 0.5, 0.0
    i = int(np.nanargmax(f1s))
    return float(thr[i]), float(f1s[i])

def compute_global_metrics(df: pd.DataFrame):
    m = {"roc_auc": float("nan"), "pr_auc": float("nan"), "brier": float("nan"), "logloss": float("nan")}
    if "y_true" in df.columns:
        y = df["y_true"].values
        p = df["proba"].values
        try: m["roc_auc"] = roc_auc_score(y, p)
        except: pass
        try: m["pr_auc"] = average_precision_score(y, p)
        except: pass
        try: m["brier"] = brier_score_loss(y, p)
        except: pass
        try:
            eps = 1e-12
            m["logloss"] = log_loss(y, np.clip(p, eps, 1 - eps))
        except: pass
    return m

def _savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()

def plot_hist(df: pd.DataFrame, path: str):
    plt.figure()
    plt.hist(df["proba"].values, bins=30)
    plt.title("Distribución de probabilidades")
    plt.xlabel("proba"); plt.ylabel("frecuencia")
    _savefig(path)

def plot_roc(df: pd.DataFrame, path: str):
    if "y_true" not in df.columns: 
        return
    fpr, tpr, _ = roc_curve(df["y_true"].values, df["proba"].values)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0,1], [0,1], linestyle="--", label="azar")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC"); plt.legend()
    _savefig(path)

def plot_pr(df: pd.DataFrame, path: str):
    if "y_true" not in df.columns:
        return
    precision, recall, _ = precision_recall_curve(df["y_true"].values, df["proba"].values)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall")
    _savefig(path)

def plot_calibration(df: pd.DataFrame, path: str):
    if "y_true" not in df.columns:
        return
    y, p = df["y_true"].values, df["proba"].values
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    plt.figure()
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.plot([0,1], [0,1], "--")
    plt.xlabel("Predicho"); plt.ylabel("Observado"); plt.title("Calibración")
    _savefig(path)

def merge_prices_probs(prices: pd.DataFrame, probs: pd.DataFrame) -> pd.DataFrame:
    a = prices.copy()
    b = probs[["date", "proba"] + (["y_true"] if "y_true" in probs.columns else [])].copy()
    a = a.sort_values("date")
    b = b.sort_values("date")
    out = pd.merge_asof(a, b, on="date", direction="nearest")
    return out

def plot_price_with_signals(prices_probs: pd.DataFrame, thr: float, path: str, title: str):
    df = prices_probs.dropna(subset=["price", "proba"]).copy()
    if df.empty:
        return
    df["pred"] = (df["proba"] >= thr).astype(int)

    df["pred_shift"] = df["pred"].shift(1).fillna(0).astype(int)
    buy_idx  = df.index[(df["pred_shift"] == 0) & (df["pred"] == 1)].tolist()
    sell_idx = df.index[(df["pred_shift"] == 1) & (df["pred"] == 0)].tolist()

    plt.figure()
    plt.plot(df["date"], df["price"])
    if buy_idx:
        plt.scatter(df.loc[buy_idx, "date"], df.loc[buy_idx, "price"], marker="^", s=50)
    if sell_idx:
        plt.scatter(df.loc[sell_idx, "date"], df.loc[sell_idx, "price"], marker="v", s=50)
    plt.title(title + f"  (thr={thr:.3f})")
    plt.xlabel("fecha"); plt.ylabel("precio")
    _savefig(path)

def confusion_at_threshold(df: pd.DataFrame, thr: float):
    if "y_true" not in df.columns:
        return None
    y = df["y_true"].values
    p = df["proba"].values
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0,1]).ravel()
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec  = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1   = (2*prec*rec)/(prec+rec) if prec + rec > 0 else 0.0
    return {"TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
            "Precision": prec, "Recall": rec, "F1": f1}

def open_browser(path: str):
    try:
        abspath = os.path.abspath(path)
        url = "file:///" + abspath.replace("\\", "/")
        webbrowser.open(url)
    except Exception as e:
        print("[make_report] No se pudo abrir el navegador:", e)

# ------------------------ HTML ------------------------
def _fmt(x):
    if x != x:
        return "-"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)

def build_html(cfg, meta, metrics, imgs, conf_cfg, conf_best, has_label):
    gen = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ticker = cfg.get("ticker", "-")
    start  = cfg.get("start_date", "-")
    source = cfg.get("data_source", "-")
    thr_cfg = cfg.get("proba_threshold", 0.5)

    css = """
    <style>
    body{font-family:Inter,system-ui,Segoe UI,Arial;margin:24px;color:#111}
    h1{margin:0 0 8px}
    .meta{color:#777;margin-bottom:16px}
    .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:16px}
    .card{border:1px solid #eee;border-radius:12px;padding:12px;box-shadow:0 4px 8px rgba(0,0,0,.04)}
    .card h3{margin:0 0 8px}
    .kv{font-size:14px;line-height:1.6}
    .muted{color:#777;font-size:12px}
    .img{width:100%;height:auto;border:1px solid #eee;border-radius:8px}
    </style>
    """

    metrics_html = f"""
      <div class="card">
        <h3>Métricas globales</h3>
        <div class="kv">
          ROC-AUC: {_fmt(metrics.get('roc_auc', float('nan')))}<br/>
          PR-AUC: {_fmt(metrics.get('pr_auc', float('nan')))}<br/>
          Brier: {_fmt(metrics.get('brier', float('nan')))}<br/>
          LogLoss: {_fmt(metrics.get('logloss', float('nan')))}<br/>
        </div>
      </div>
    """

    conf_html = ""
    if has_label:
        conf_html = f"""
        <div class="card">
          <h3>Confusión (thr=cfg)</h3>
          <div class="kv">
            TP: {_fmt(conf_cfg['TP'])} &nbsp; FP: {_fmt(conf_cfg['FP'])}<br/>
            TN: {_fmt(conf_cfg['TN'])} &nbsp; FN: {_fmt(conf_cfg['FN'])}<br/>
            Precision: {_fmt(conf_cfg['Precision'])}<br/>
            Recall: {_fmt(conf_cfg['Recall'])}<br/>
            F1: {_fmt(conf_cfg['F1'])}<br/>
          </div>
        </div>
        <div class="card">
          <h3>Confusión (thr=bestF1)</h3>
          <div class="kv">
            TP: {_fmt(conf_best['TP'])} &nbsp; FP: {_fmt(conf_best['FP'])}<br/>
            TN: {_fmt(conf_best['TN'])} &nbsp; FN: {_fmt(conf_best['FN'])}<br/>
            Precision: {_fmt(conf_best['Precision'])}<br/>
            Recall: {_fmt(conf_best['Recall'])}<br/>
            F1: {_fmt(conf_best['F1'])}<br/>
          </div>
        </div>
        """

    def _img(tag):
        p = imgs.get(tag)
        return (f'<img class="img" src="img/{os.path.basename(p)}" />' if p and os.path.isfile(p) else
                '<div class="muted">no disponible</div>')

    html = f"""<!doctype html>
<html lang="es">
<head><meta charset="utf-8"><title>Reporte ML / BvB</title>{css}</head>
<body>
  <h1>Reporte Técnico</h1>
  <div class="meta">Generado: {gen}</div>

  <div class="card">
    <h3>Config (detectada)</h3>
    <div class="kv">
      Ticker: {ticker}<br/>
      Start: {start}<br/>
      Fuente: {source}<br/>
      Prob. threshold (cfg): {_fmt(thr_cfg)}<br/>
      Label source: {('probabilities_ml.csv' if has_label else '<span class="muted">no se encontró y_true</span>')}
    </div>
  </div>

  <div class="grid">
    {metrics_html}
    {conf_html}
  </div>

  <div class="grid">
    <div class="card"><h3>Histograma de probabilidades</h3>{_img('hist')}</div>
    <div class="card"><h3>Curva ROC</h3>{_img('roc')}</div>
    <div class="card"><h3>Curva PR</h3>{_img('pr')}</div>
    <div class="card"><h3>Calibración</h3>{_img('calib')}</div>
  </div>

  <div class="grid">
    <div class="card"><h3>Precio + señales (thr=cfg)</h3>{_img('px_cfg')}</div>
    <div class="card"><h3>Precio + señales (thr=bestF1)</h3>{_img('px_best')}</div>
  </div>

  <div class="card">
    <h3>Notas</h3>
    <ul class="muted">
      <li>Si quieres métricas completas y matrices de confusión, guarda <code>y_true</code> junto con <code>proba</code> en <code>probabilities_ml.csv</code>.</li>
      <li>Las señales BUY/SELL marcan cambios de estado de predicción (0→1 y 1→0) sobre el precio.</li>
    </ul>
  </div>
</body>
</html>
"""
    return html

# ------------------------ main ------------------------
def main():
    print("[make_report] Leyendo artifacts...")
    _ensure_dirs()

    cfg, meta = _read_summary()
    try:
        probs, meta = smart_read_probabilities()
        probs, _clean_info = clean_probs(probs)

    except Exception as e:
        with open(HTML_OUT, "w", encoding="utf-8") as f:
            f.write("""<!doctype html><meta charset="utf-8">
            <h1>Reporte LITE</h1>
            <p>No se pudo leer <code>artifacts/probabilities_ml.csv</code>.</p>""")
        print("[make_report] No se pudo leer probabilities_ml.csv:", e)
        open_browser(HTML_OUT)
        return

    thr_cfg = float(cfg.get("proba_threshold", 0.5))

    metrics = compute_global_metrics(probs)

    has_label = "y_true" in probs.columns
    thr_best, f1_best = (thr_cfg, float("nan"))
    conf_cfg = conf_best = None
    if has_label:
        thr_best, f1_best = best_f1_threshold(probs["y_true"].values, probs["proba"].values)
        conf_cfg  = confusion_at_threshold(probs, thr_cfg)
        conf_best = confusion_at_threshold(probs, thr_best)

    imgs = {}
    imgs["hist"] = os.path.join(IMG_DIR, "hist.png")
    plot_hist(probs, imgs["hist"])

    imgs["roc"] = os.path.join(IMG_DIR, "roc.png");     plot_roc(probs, imgs["roc"])
    imgs["pr"]  = os.path.join(IMG_DIR, "pr.png");      plot_pr(probs, imgs["pr"])
    imgs["calib"]=os.path.join(IMG_DIR, "calib.png");   plot_calibration(probs, imgs["calib"])

    prices = smart_read_prices(cfg)
    if prices is not None:
        pp = merge_prices_probs(prices, probs)
        imgs["px_cfg"]  = os.path.join(IMG_DIR, "price_thr_cfg.png")
        plot_price_with_signals(pp, thr_cfg, imgs["px_cfg"], "Señales (cfg)")

        imgs["px_best"] = os.path.join(IMG_DIR, "price_thr_bestf1.png")
        plot_price_with_signals(pp, thr_best, imgs["px_best"], "Señales (best F1)")
    else:
        imgs["px_cfg"] = imgs["px_best"] = None

    html = build_html(cfg, meta, metrics, imgs, conf_cfg, conf_best, has_label)
    with open(HTML_OUT, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[make_report] HTML listo → {HTML_OUT}")
    open_browser(HTML_OUT)

if __name__ == "__main__":
    main()
