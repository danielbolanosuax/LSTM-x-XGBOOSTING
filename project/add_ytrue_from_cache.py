# project/add_ytrue_from_cache.py
import json
from pathlib import Path
import sys
import pandas as pd
import numpy as np

ART = Path("artifacts")
BVB = Path("bvb")
DATA = Path("data")
PROB_CSV = ART / "probabilities_ml.csv"
SUMMARY_JSON = BVB / "summary_ml.json"

def _read_probs(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}")
    # intentos robustos para detectar la columna temporal
    df = pd.read_csv(path)
    # normaliza nombres
    cols_lower = {c: c.lower() for c in df.columns}
    df.columns = list(cols_lower.values())

    date_col = None
    for c in ("date","datetime","time","index","Unnamed: 0".lower()):
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        # último recurso: si no hay columna temporal, crea índice incremental
        df["date"] = np.arange(len(df))
        date_col = "date"

    # parsea fechas si parecen fechas
    try:
        df[date_col] = pd.to_datetime(df[date_col])
    except Exception:
        pass

    # estandariza nombre
    if date_col != "date":
        df.rename(columns={date_col: "date"}, inplace=True)

    # normaliza nombre de probas
    if "proba" not in df.columns and "prob" in df.columns:
        df.rename(columns={"prob": "proba"}, inplace=True)

    return df

def _guess_cache_name(cfg: dict) -> Path:
    tkr = cfg.get("ticker","AAPL")
    start = (cfg.get("start_date") or "2016-01-01").replace("-","")
    use_adj = bool(cfg.get("use_adjusted_close", True))
    suffix = "adj" if use_adj else "raw"
    # igual que el log: AAPL_20160101_None_adj.csv
    fname = f"{tkr}_{start}_None_{suffix}.csv"
    return DATA / fname

def _read_price_series(cache_path: Path) -> pd.DataFrame:
    if not cache_path.exists():
        raise FileNotFoundError(f"No encuentro cache OHLC {cache_path}")
    df = pd.read_csv(cache_path)
    # intenta detectar columna de fecha
    for c in ("Date","date","datetime","Datetime","time","Time"):
        if c in df.columns:
            try:
                df[c] = pd.to_datetime(df[c])
                df.rename(columns={c:"date"}, inplace=True)
                break
            except Exception:
                pass
    if "date" not in df.columns:
        # si no hay fecha, usa índice incremental
        df["date"] = pd.RangeIndex(len(df))
    # columna de precio
    price_col = "Adj Close" if "Adj Close" in df.columns else ("Close" if "Close" in df.columns else None)
    if price_col is None:
        # si no hay, intenta algo razonable
        numeric_cols = [c for c in df.columns if c.lower() in ("close","adjclose","adj_close") or df[c].dtype.kind in "fi"]
        if not numeric_cols:
            raise ValueError("No encuentro columna de precio (Close / Adj Close).")
        price_col = numeric_cols[0]
    df = df[["date", price_col]].rename(columns={price_col:"close"})
    return df

def _make_labels(close: pd.Series, horizon: int) -> pd.Series:
    """
    Etiqueta binaria simple por dirección a h pasos:
    y_true = 1 si retorno futuro a h días > 0, si no 0.
    """
    fut_ret = close.shift(-horizon) / close - 1.0
    y = (fut_ret > 0).astype("Int64")
    return y

def main():
    # 1) lee probs
    probs = _read_probs(PROB_CSV)

    # 2) lee configuración (horizon, ticker, etc.)
    if SUMMARY_JSON.exists():
        cfg = json.load(open(SUMMARY_JSON, "r", encoding="utf-8")).get("config", {})
    else:
        cfg = {}
    horizon = int(cfg.get("horizon", 20))

    # 3) abre cache OHLC
    cache_path = _guess_cache_name(cfg)
    price_df = _read_price_series(cache_path)

    # 4) calcula y_true y lo alinea por fecha
    labels = _make_labels(price_df["close"], horizon)
    lab_df = pd.DataFrame({"date": price_df["date"], "y_true": labels})

    # 5) merge robusto
    merged = probs.copy()
    if pd.api.types.is_datetime64_any_dtype(lab_df["date"]) and pd.api.types.is_datetime64_any_dtype(merged["date"]):
        on = "date"
        merged = pd.merge(merged, lab_df, on=on, how="left")
    else:
        # fallback: alinea por cola (mismo largo) si el merge por fecha no cuadra
        min_len = min(len(merged), len(lab_df))
        merged = merged.tail(min_len).reset_index(drop=True)
        lab_tail = lab_df.tail(min_len).reset_index(drop=True)
        merged["y_true"] = lab_tail["y_true"]

    # 6) guarda
    merged.to_csv(PROB_CSV, index=False)
    n_ok = int(merged["y_true"].notna().sum())
    print(f"[add_ytrue] Etiquetas añadidas en probabilities_ml.csv -> {n_ok} filas con y_true (h={horizon}).")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[add_ytrue] ERROR: {e}")
        sys.exit(1)
