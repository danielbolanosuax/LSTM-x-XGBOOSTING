# path: project/av_client.py
"""
Cliente Alpha Vantage con endpoints principales, retries/backoff y normalización a OHLCV.

Por qué: evitar dependencias pesadas, soportar fallback y tolerar rate-limits.
Doc oficial (nombres de funciones/params/series): https://www.alphavantage.co/documentation/
"""

from __future__ import annotations
import os
import time
import csv
import io
import json
import argparse
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any, Iterable

import requests
import pandas as pd


# -------- util sesión --------
def _default_session(timeout: int = 30) -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "av-client/1.0 (+alpha)"})
    s.timeout = timeout  # type: ignore[attr-defined]
    # Razón: asegurar certs válidos incluso en Windows con rutas “raras”
    try:
        import certifi
        s.verify = certifi.where()
    except Exception:
        pass
    return s


# -------- config del cliente --------
@dataclass
class AVConfig:
    apikey: str
    retries: int = 5
    backoff_sec: int = 15  # Alpha Vantage rate-limit -> esperar y reintentar
    base_url: str = "https://www.alphavantage.co/query"
    datatype_default: str = "json"  # "json" o "csv"


class AlphaVantageClient:
    def __init__(self, cfg: AVConfig, session: Optional[requests.Session] = None) -> None:
        self.cfg = cfg
        self.s = session or _default_session()

    # --- Request con backoff y manejo de mensajes de cuota/premium ---
    def _request(self, params: Dict[str, Any], datatype: Optional[str] = None) -> Tuple[str, bytes]:
        q = dict(params)
        q["apikey"] = self.cfg.apikey
        dt = (datatype or self.cfg.datatype_default).lower()
        if dt not in ("json", "csv"):
            dt = "json"
        q["datatype"] = dt

        last_exc = None
        for i in range(self.cfg.retries):
            try:
                r = self.s.get(self.cfg.base_url, params=q, timeout=60)
                r.raise_for_status()
                content = r.content
                # Si es JSON, mirar mensajes de cuota/premium
                if dt == "json":
                    txt = content.decode("utf-8", errors="ignore")
                    try:
                        data = json.loads(txt)
                    except Exception:
                        # A veces la API responde HTML si hay mantenimiento
                        if i < self.cfg.retries - 1:
                            time.sleep(self.cfg.backoff_sec)
                            continue
                        return dt, content

                    # Mensajes típicos de límite/premium
                    note = str(data.get("Note", "")).lower()
                    info = str(data.get("Information", "")).lower()
                    errm = str(data.get("Error Message", "")).lower()
                    if "premium" in note or "premium" in info or "premium" in errm:
                        # Devolver tal cual para que la capa superior pueda elegir fallback
                        return dt, content
                    if note or info:
                        # Rate limit: esperar y reintentar
                        if i < self.cfg.retries - 1:
                            time.sleep(self.cfg.backoff_sec)
                            continue
                    return dt, content

                # CSV: devolver bytes
                return dt, content
            except Exception as e:
                last_exc = e
                if i < self.cfg.retries - 1:
                    time.sleep(self.cfg.backoff_sec)
                    continue
                raise

        # Nunca debería llegar aquí
        if last_exc:
            raise last_exc
        raise RuntimeError("AlphaVantage request failed unexpectedly")

    # ---------- Core Time Series (acciones) ----------
    # Doc: TIME_SERIES_DAILY, TIME_SERIES_DAILY_ADJUSTED, WEEKLY(_ADJUSTED), MONTHLY(_ADJUSTED), TIME_SERIES_INTRADAY. :contentReference[oaicite:2]{index=2}
    def time_series_daily(self, symbol: str, adjusted: bool = True, outputsize: str = "full", datatype: str = "json") -> Tuple[str, bytes]:
        fn = "TIME_SERIES_DAILY_ADJUSTED" if adjusted else "TIME_SERIES_DAILY"
        return self._request({"function": fn, "symbol": symbol, "outputsize": outputsize}, datatype)

    def time_series_weekly(self, symbol: str, adjusted: bool = False, datatype: str = "json") -> Tuple[str, bytes]:
        fn = "TIME_SERIES_WEEKLY_ADJUSTED" if adjusted else "TIME_SERIES_WEEKLY"
        return self._request({"function": fn, "symbol": symbol}, datatype)

    def time_series_monthly(self, symbol: str, adjusted: bool = False, datatype: str = "json") -> Tuple[str, bytes]:
        fn = "TIME_SERIES_MONTHLY_ADJUSTED" if adjusted else "TIME_SERIES_MONTHLY"
        return self._request({"function": fn, "symbol": symbol}, datatype)

    def time_series_intraday(
        self,
        symbol: str,
        interval: str = "5min",
        outputsize: str = "compact",
        adjusted: bool = True,
        extended_hours: bool = True,
        month: Optional[str] = None,
        datatype: str = "json",
    ) -> Tuple[str, bytes]:
        # Params según doc (interval obligatorio; adjusted/extended_hours opcional). :contentReference[oaicite:3]{index=3}
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
            "adjusted": str(adjusted).lower(),
            "extended_hours": str(extended_hours).lower(),
        }
        if month:
            params["month"] = month
        return self._request(params, datatype)

    # ---------- Utilidades ----------
    # Doc: GLOBAL_QUOTE, SYMBOL_SEARCH. :contentReference[oaicite:4]{index=4}
    def global_quote(self, symbol: str, datatype: str = "json") -> Tuple[str, bytes]:
        return self._request({"function": "GLOBAL_QUOTE", "symbol": symbol}, datatype)

    def symbol_search(self, keywords: str, datatype: str = "json") -> Tuple[str, bytes]:
        return self._request({"function": "SYMBOL_SEARCH", "keywords": keywords}, datatype)

    # ---------- Indicadores Técnicos ----------
    # Doc: RSI, MACD, STOCH. :contentReference[oaicite:5]{index=5}
    def rsi(self, symbol: str, interval: str = "daily", time_period: int = 14, series_type: str = "close", datatype: str = "json"):
        return self._request({"function": "RSI", "symbol": symbol, "interval": interval, "time_period": time_period, "series_type": series_type}, datatype)

    def macd(self, symbol: str, interval: str = "daily", series_type: str = "close", fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9, datatype: str = "json"):
        return self._request({
            "function": "MACD", "symbol": symbol, "interval": interval, "series_type": series_type,
            "fastperiod": fastperiod, "slowperiod": slowperiod, "signalperiod": signalperiod
        }, datatype)

    def stoch(self, symbol: str, interval: str = "daily", fastkperiod: int = 14, slowkperiod: int = 3, slowdperiod: int = 3, datatype: str = "json"):
        return self._request({
            "function": "STOCH", "symbol": symbol, "interval": interval,
            "fastkperiod": fastkperiod, "slowkperiod": slowkperiod, "slowdperiod": slowdperiod
        }, datatype)

    # ---------- FX ----------
    # Doc: FX_DAILY, CURRENCY_EXCHANGE_RATE. :contentReference[oaicite:6]{index=6}
    def fx_daily(self, from_symbol: str, to_symbol: str, outputsize: str = "full", datatype: str = "json"):
        return self._request({"function": "FX_DAILY", "from_symbol": from_symbol, "to_symbol": to_symbol, "outputsize": outputsize}, datatype)

    def currency_exchange_rate(self, from_currency: str, to_currency: str, datatype: str = "json"):
        return self._request({"function": "CURRENCY_EXCHANGE_RATE", "from_currency": from_currency, "to_currency": to_currency}, datatype)

    # ---------- Cripto ----------
    # Doc: DIGITAL_CURRENCY_DAILY (el más útil para histórico). :contentReference[oaicite:7]{index=7}
    def digital_currency_daily(self, symbol: str, market: str = "USD", datatype: str = "json"):
        return self._request({"function": "DIGITAL_CURRENCY_DAILY", "symbol": symbol, "market": market}, datatype)

    # ---------- Fundamentals ----------
    # Doc: OVERVIEW, INCOME_STATEMENT, BALANCE_SHEET, CASH_FLOW. :contentReference[oaicite:8]{index=8}
    def overview(self, symbol: str, datatype: str = "json"):
        return self._request({"function": "OVERVIEW", "symbol": symbol}, datatype)

    def income_statement(self, symbol: str, datatype: str = "json"):
        return self._request({"function": "INCOME_STATEMENT", "symbol": symbol}, datatype)

    def balance_sheet(self, symbol: str, datatype: str = "json"):
        return self._request({"function": "BALANCE_SHEET", "symbol": symbol}, datatype)

    def cash_flow(self, symbol: str, datatype: str = "json"):
        return self._request({"function": "CASH_FLOW", "symbol": symbol}, datatype)


# -------- helpers de parseo a pandas --------
def _json_or_raise(dt: str, blob: bytes) -> Dict[str, Any]:
    if dt != "json":
        raise ValueError("Se esperaba JSON")
    data = json.loads(blob.decode("utf-8", errors="ignore"))
    if "Error Message" in data:
        raise RuntimeError(data["Error Message"])
    return data

def _first_dict_like(data: Dict[str, Any], keys_hint: Iterable[str]) -> Tuple[str, Dict[str, Any]]:
    for k in data.keys():
        for h in keys_hint:
            if h.lower() in k.lower():
                node = data[k]
                if isinstance(node, dict):
                    return k, node
    raise KeyError(f"No se encontró ninguna clave de series con hints: {list(keys_hint)}")

def _parse_time_series_ohlcv(data: Dict[str, Any], prefer_adjusted: bool = True) -> pd.DataFrame:
    # Acciones Daily/Weekly/Monthly/Intraday
    # Busca la primera clave que contenga "Time Series"
    _, series = _first_dict_like(data, ["Time Series"])
    # Mapear columnas numéricas; daily_adjusted trae "5. adjusted close"
    rows = []
    for ts, row in series.items():
        # claves numéricas como "1. open", ...
        open_ = row.get("1. open") or row.get("1. Open") or row.get("1. open ")
        high_ = row.get("2. high")
        low_  = row.get("3. low")
        close_adj = row.get("5. adjusted close")
        close_raw = row.get("4. close")
        volume = row.get("6. volume") or row.get("5. volume")
        close = close_adj if (prefer_adjusted and close_adj is not None) else close_raw
        if close is None:
            close = close_raw
        rows.append((ts, open_, high_, low_, close, volume))
    df = pd.DataFrame(rows, columns=["ts","Open","High","Low","Close","Volume"])
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts").set_index("ts")
    for c in ["Open","High","Low","Close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "Volume" in df:
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    return df.dropna(subset=["Open","High","Low","Close"])

def _parse_fx_daily(data: Dict[str, Any]) -> pd.DataFrame:
    # FX_DAILY: clave "Time Series FX (Daily)" con "1. open"... sin volumen
    _, series = _first_dict_like(data, ["Time Series FX"])
    rows = []
    for ts, row in series.items():
        rows.append((ts, row.get("1. open"), row.get("2. high"), row.get("3. low"), row.get("4. close")))
    df = pd.DataFrame(rows, columns=["ts","Open","High","Low","Close"])
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts").set_index("ts")
    for c in ["Open","High","Low","Close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Volume"] = pd.NA
    return df.dropna(subset=["Open","High","Low","Close"])

def _parse_crypto_daily(data: Dict[str, Any], market: str = "USD") -> pd.DataFrame:
    # DIGITAL_CURRENCY_DAILY: claves "1a. open (USD)", "1b. open (USD)"...
    _, series = _first_dict_like(data, ["Digital Currency Daily"])
    m = market.upper()
    rows = []
    for ts, row in series.items():
        open_ = row.get(f"1a. open ({m})") or row.get(f"1b. open ({m})")
        high_ = row.get(f"2a. high ({m})") or row.get(f"2b. high ({m})")
        low_  = row.get(f"3a. low ({m})")  or row.get(f"3b. low ({m})")
        close_= row.get(f"4a. close ({m})") or row.get(f"4b. close ({m})")
        vol   = row.get("5. volume")
        rows.append((ts, open_, high_, low_, close_, vol))
    df = pd.DataFrame(rows, columns=["ts","Open","High","Low","Close","Volume"])
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts").set_index("ts")
    for c in ["Open","High","Low","Close","Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["Open","High","Low","Close"])

def to_ohlcv_df(dt: str, blob: bytes, series_kind: str = "equity", prefer_adjusted: bool = True, market: str = "USD") -> pd.DataFrame:
    if dt == "csv":
        df = pd.read_csv(io.BytesIO(blob))
        # Intento de normalización simple si viene CSV crudo de intraday
        cols = {c.lower(): c for c in df.columns}
        for need in ["open","high","low","close"]:
            if need not in cols:
                raise RuntimeError("CSV no reconocido; usa datatype=json para autoinferir")
        df.columns = [c.title() for c in df.columns]
        if "Datetime" in df.columns:
            df = df.rename(columns={"Datetime":"ts"}).set_index(pd.to_datetime(df["ts"])).drop(columns=["ts"])
        elif "Timestamp" in df.columns:
            df = df.rename(columns={"Timestamp":"ts"}).set_index(pd.to_datetime(df["ts"])).drop(columns=["ts"])
        return df.sort_index()
    data = _json_or_raise(dt, blob)
    if series_kind == "equity":
        return _parse_time_series_ohlcv(data, prefer_adjusted=prefer_adjusted)
    if series_kind == "fx":
        return _parse_fx_daily(data)
    if series_kind == "crypto":
        return _parse_crypto_daily(data, market=market)
    raise ValueError(f"series_kind desconocido: {series_kind}")


# -------- CLI --------
def _save(df: pd.DataFrame, out: Optional[str], parquet: bool) -> None:
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        if parquet:
            df.to_parquet(out if out.endswith(".parquet") else out + ".parquet")
        else:
            df.to_csv(out if out.endswith(".csv") else out + ".csv")
    else:
        # por qué: permitir piping rápido
        df.to_csv(sys.stdout)

def _cli_build() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Alpha Vantage CLI")
    p.add_argument("--apikey", default=os.getenv("ALPHA_VANTAGE_KEY", ""), help="API key (o env ALPHA_VANTAGE_KEY)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # acciones
    sp = sub.add_parser("daily")
    sp.add_argument("--symbol", required=True)
    sp.add_argument("--adjusted", action="store_true")
    sp.add_argument("--outputsize", default="full", choices=["full","compact"])
    sp.add_argument("--datatype", default="json", choices=["json","csv"])
    sp.add_argument("--out")

    sp = sub.add_parser("intraday")
    sp.add_argument("--symbol", required=True)
    sp.add_argument("--interval", default="5min", choices=["1min","5min","15min","30min","60min"])
    sp.add_argument("--outputsize", default="compact", choices=["compact","full"])
    sp.add_argument("--adjusted", action="store_true")
    sp.add_argument("--extended-hours", dest="extended_hours", action="store_true")
    sp.add_argument("--month")
    sp.add_argument("--datatype", default="json", choices=["json","csv"])
    sp.add_argument("--out")

    sp = sub.add_parser("weekly")
    sp.add_argument("--symbol", required=True)
    sp.add_argument("--adjusted", action="store_true")
    sp.add_argument("--datatype", default="json", choices=["json","csv"])
    sp.add_argument("--out")

    sp = sub.add_parser("monthly")
    sp.add_argument("--symbol", required=True)
    sp.add_argument("--adjusted", action="store_true")
    sp.add_argument("--datatype", default="json", choices=["json","csv"])
    sp.add_argument("--out")

    # util
    sp = sub.add_parser("quote")
    sp.add_argument("--symbol", required=True)

    sp = sub.add_parser("search")
    sp.add_argument("--keywords", required=True)

    # TI
    sp = sub.add_parser("ti-rsi")
    sp.add_argument("--symbol", required=True)
    sp.add_argument("--interval", default="daily")
    sp.add_argument("--time-period", type=int, default=14)
    sp.add_argument("--series-type", default="close")

    sp = sub.add_parser("ti-macd")
    sp.add_argument("--symbol", required=True)
    sp.add_argument("--interval", default="daily")
    sp.add_argument("--series-type", default="close")
    sp.add_argument("--fast", type=int, default=12)
    sp.add_argument("--slow", type=int, default=26)
    sp.add_argument("--signal", type=int, default=9)

    sp = sub.add_parser("ti-stoch")
    sp.add_argument("--symbol", required=True)
    sp.add_argument("--interval", default="daily")
    sp.add_argument("--fastk", type=int, default=14)
    sp.add_argument("--slowk", type=int, default=3)
    sp.add_argument("--slowd", type=int, default=3)

    # FX
    sp = sub.add_parser("fx-daily")
    sp.add_argument("--from", dest="from_symbol", required=True)
    sp.add_argument("--to", dest="to_symbol", required=True)
    sp.add_argument("--outputsize", default="full", choices=["full","compact"])
    sp.add_argument("--datatype", default="json", choices=["json","csv"])
    sp.add_argument("--out")

    sp = sub.add_parser("fx-rate")
    sp.add_argument("--from", dest="from_currency", required=True)
    sp.add_argument("--to", dest="to_currency", required=True)

    # Crypto
    sp = sub.add_parser("crypto-daily")
    sp.add_argument("--symbol", required=True)
    sp.add_argument("--market", default="USD")
    sp.add_argument("--datatype", default="json", choices=["json","csv"])
    sp.add_argument("--out")

    # Fundamentals
    sub.add_parser("overview").add_argument("--symbol", required=True)
    sub.add_parser("income").add_argument("--symbol", required=True)
    sub.add_parser("balance").add_argument("--symbol", required=True)
    sub.add_parser("cashflow").add_argument("--symbol", required=True)

    return p

def main() -> None:
    import sys
    args = _cli_build().parse_args()
    if not args.apikey:
        print("Falta --apikey o env ALPHA_VANTAGE_KEY")
        raise SystemExit(2)

    cli = AlphaVantageClient(AVConfig(apikey=args.apikey))
    # acciones
    if args.cmd == "daily":
        dt, blob = cli.time_series_daily(args.symbol, adjusted=args.adjusted, outputsize=args.outputsize, datatype=args.datatype)
        df = to_ohlcv_df(dt, blob, series_kind="equity", prefer_adjusted=args.adjusted)
        if args.out: _save(df, args.out, parquet=False)
        else: print(df.tail().to_string())
        return
    if args.cmd == "intraday":
        dt, blob = cli.time_series_intraday(args.symbol, interval=args.interval, outputsize=args.outputsize, adjusted=args.adjusted, extended_hours=args.extended_hours, month=args.month, datatype=args.datatype)
        df = to_ohlcv_df(dt, blob, series_kind="equity", prefer_adjusted=args.adjusted)
        if args.out: _save(df, args.out, parquet=False)
        else: print(df.tail().to_string())
        return
    if args.cmd == "weekly":
        dt, blob = cli.time_series_weekly(args.symbol, adjusted=args.adjusted, datatype=args.datatype)
        df = to_ohlcv_df(dt, blob, series_kind="equity", prefer_adjusted=args.adjusted)
        if args.out: _save(df, args.out, parquet=False)
        else: print(df.tail().to_string()); return
    if args.cmd == "monthly":
        dt, blob = cli.time_series_monthly(args.symbol, adjusted=args.adjusted, datatype=args.datatype)
        df = to_ohlcv_df(dt, blob, series_kind="equity", prefer_adjusted=args.adjusted)
        if args.out: _save(df, args.out, parquet=False)
        else: print(df.tail().to_string()); return

    # util
    if args.cmd == "quote":
        dt, blob = cli.global_quote(args.symbol)
        print(blob.decode("utf-8", errors="ignore")); return
    if args.cmd == "search":
        dt, blob = cli.symbol_search(args.keywords)
        print(blob.decode("utf-8", errors="ignore")); return

    # TI
    if args.cmd == "ti-rsi":
        dt, blob = cli.rsi(args.symbol, interval=args.interval, time_period=args.time_period, series_type=args.series_type)
        print(blob.decode("utf-8", errors="ignore")); return
    if args.cmd == "ti-macd":
        dt, blob = cli.macd(args.symbol, interval=args.interval, series_type=args.series_type, fastperiod=args.fast, slowperiod=args.slow, signalperiod=args.signal)
        print(blob.decode("utf-8", errors="ignore")); return
    if args.cmd == "ti-stoch":
        dt, blob = cli.stoch(args.symbol, interval=args.interval, fastkperiod=args.fastk, slowkperiod=args.slowk, slowdperiod=args.slowd)
        print(blob.decode("utf-8", errors="ignore")); return

    # FX
    if args.cmd == "fx-daily":
        dt, blob = cli.fx_daily(args.from_symbol, args.to_symbol, outputsize=args.outputsize, datatype=args.datatype)
        df = to_ohlcv_df(dt, blob, series_kind="fx")
        if args.out: _save(df, args.out, parquet=False)
        else: print(df.tail().to_string()); return
    if args.cmd == "fx-rate":
        dt, blob = cli.currency_exchange_rate(args.from_currency, args.to_currency)
        print(blob.decode("utf-8", errors="ignore")); return

    # Crypto
    if args.cmd == "crypto-daily":
        dt, blob = cli.digital_currency_daily(args.symbol, market=args.market, datatype=args.datatype)
        df = to_ohlcv_df(dt, blob, series_kind="crypto", market=args.market)
        if args.out: _save(df, args.out, parquet=False)
        else: print(df.tail().to_string()); return

    # Fundamentals
    if args.cmd == "overview":
        dt, blob = cli.overview(args.symbol); print(blob.decode("utf-8", errors="ignore")); return
    if args.cmd == "income":
        dt, blob = cli.income_statement(args.symbol); print(blob.decode("utf-8", errors="ignore")); return
    if args.cmd == "balance":
        dt, blob = cli.balance_sheet(args.symbol); print(blob.decode("utf-8", errors="ignore")); return
    if args.cmd == "cashflow":
        dt, blob = cli.cash_flow(args.symbol); print(blob.decode("utf-8", errors="ignore")); return


if __name__ == "__main__":
    main()
