# backtest/data.py
# Téléchargement de l'historique de bougies.
# Sources essayées dans l'ordre (source="auto") :
#   1) data.binance.vision : fichiers officiels Binance (prix exacts de Binance, sans limite)
#   2) KuCoin  (API publique, par tranches de 1500 bougies)
#   3) Bybit   (API publique, par tranches de 1000 bougies)
# Résultat : DataFrame avec les colonnes time (UTC), open, high, low, close, volume.

import io
import os
import time
import zipfile
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import requests

SECONDS = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400}
KUCOIN_TYPES = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
                "1h": "1hour", "4h": "4hour", "1d": "1day"}
BYBIT_INTERVALS = {"1m": "1", "5m": "5", "15m": "15", "30m": "30", "1h": "60", "4h": "240", "1d": "D"}
COLS = ["time", "open", "high", "low", "close", "volume"]
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def _parse_date(s):
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def _frame(rows):
    """rows : liste de [ts_ms, open, high, low, close, volume]"""
    df = pd.DataFrame(rows, columns=COLS)
    df["time"] = pd.to_datetime(df["time"].astype("int64"), unit="ms", utc=True)
    for c in COLS[1:]:
        df[c] = df[c].astype(float)
    return df.drop_duplicates("time").sort_values("time").reset_index(drop=True)


def _get(url, params=None, retries=3):
    """GET avec quelques essais ; renvoie la réponse, ou None si 404 / erreur réseau."""
    for k in range(retries):
        try:
            r = requests.get(url, params=params, timeout=30)
        except requests.RequestException as e:
            print(f"  réseau: {e}")
            time.sleep(1 + k)
            continue
        if r.status_code == 200:
            return r
        if r.status_code == 404:
            return None
        if r.status_code in (429, 418) or r.status_code >= 500:
            time.sleep(2 * (k + 1))
            continue
        return None
    return None


# ---------------------------------------------------------------- Binance (data.binance.vision)

def _parse_vision_zip(content):
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f, header=None, usecols=range(6))
    ts = pd.to_numeric(df[0], errors="coerce")
    df = df[ts.notna()].copy()
    ts = ts[ts.notna()].astype("int64")
    ts = ts.where(ts < 10**14, ts // 1000)  # certains fichiers récents sont en microsecondes
    df[0] = ts
    return df.values.tolist()


def fetch_binance_vision(symbol, interval, start_ms, end_ms):
    base = "https://data.binance.vision/data/spot"
    today = datetime.now(timezone.utc).date()
    first_this_month = today.replace(day=1)
    prev_month_start = (first_this_month - timedelta(days=1)).replace(day=1)
    start_d = datetime.fromtimestamp(start_ms / 1000, timezone.utc).date()
    end_d = datetime.fromtimestamp(end_ms / 1000, timezone.utc).date()
    rows = []
    cur = start_d.replace(day=1)
    while cur <= end_d:
        nxt = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)
        got = None
        if cur < first_this_month:  # mois terminé : fichier mensuel
            url = f"{base}/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{cur:%Y-%m}.zip"
            r = _get(url)
            if r is not None:
                got = _parse_vision_zip(r.content)
        if got is not None:
            rows.extend(got)
        elif cur >= prev_month_start:  # mois récent : fichiers journaliers (le mensuel peut ne pas être publié)
            day = cur
            while day < nxt and day < today:
                if start_d <= day <= end_d:
                    url = f"{base}/daily/klines/{symbol}/{interval}/{symbol}-{interval}-{day:%Y-%m-%d}.zip"
                    r = _get(url)
                    if r is not None:
                        rows.extend(_parse_vision_zip(r.content))
                day += timedelta(days=1)
        cur = nxt
    return rows


# ---------------------------------------------------------------- KuCoin

def fetch_kucoin(symbol, interval, start_ms, end_ms):
    sym = symbol[:-4] + "-" + symbol[-4:]  # BTCUSDT -> BTC-USDT
    step = SECONDS[interval]
    start_s, cur_end = start_ms // 1000, end_ms // 1000
    rows = []
    while cur_end > start_s:
        cur_start = max(start_s, cur_end - 1500 * step)
        r = _get("https://api.kucoin.com/api/v1/market/candles",
                 {"type": KUCOIN_TYPES[interval], "symbol": sym, "startAt": cur_start, "endAt": cur_end})
        data = (r.json().get("data") if r is not None else None) or []
        for t, o, c, h, l, v, _ in data:  # KuCoin : time, open, close, high, low, volume, turnover
            rows.append([int(t) * 1000, o, h, l, c, v])
        cur_end = cur_start
        time.sleep(0.15)
    return rows


# ---------------------------------------------------------------- Bybit

def fetch_bybit(symbol, interval, start_ms, end_ms):
    step = SECONDS[interval] * 1000
    cur_end = end_ms
    rows = []
    while cur_end > start_ms:
        cur_start = max(start_ms, cur_end - 1000 * step)
        r = _get("https://api.bybit.com/v5/market/kline",
                 {"category": "spot", "symbol": symbol, "interval": BYBIT_INTERVALS[interval],
                  "start": cur_start, "end": cur_end, "limit": 1000})
        data = []
        if r is not None:
            data = (r.json().get("result") or {}).get("list") or []
        for t, o, h, l, c, v, _ in data:  # Bybit : start, open, high, low, close, volume, turnover
            rows.append([int(t), o, h, l, c, v])
        cur_end = cur_start
        time.sleep(0.15)
    return rows


SOURCES = {"binance_vision": fetch_binance_vision, "kucoin": fetch_kucoin, "bybit": fetch_bybit}


def load(symbol, interval, start, end=None, source="auto", use_cache=True):
    """Renvoie (DataFrame, nom_de_la_source). start/end au format 'YYYY-MM-DD' (end exclu, défaut = aujourd'hui)."""
    if interval not in SECONDS:
        raise ValueError(f"Unité de temps inconnue: {interval} (choix: {', '.join(SECONDS)})")
    start_ms = int(_parse_date(start).timestamp() * 1000)
    end_dt = _parse_date(end) if end else datetime.now(timezone.utc)
    end_ms = int(end_dt.timestamp() * 1000)
    if end_ms <= start_ms:
        raise ValueError("La date de fin doit être après la date de début.")

    names = list(SOURCES) if source == "auto" else [source]
    for name in names:
        cache = os.path.join(CACHE_DIR, f"{symbol}_{interval}_{start}_{end or 'now'}_{name}.csv")
        if use_cache and os.path.exists(cache) and end:
            df = pd.read_csv(cache, parse_dates=["time"])
            print(f"{symbol} {interval}: {len(df)} bougies (cache, {name})")
            return df, name
        print(f"{symbol} {interval}: téléchargement via {name}...")
        try:
            rows = SOURCES[name](symbol, interval, start_ms, end_ms)
        except Exception as e:  # une source en panne ne doit pas bloquer les autres
            print(f"  échec {name}: {e}")
            continue
        if not rows:
            print(f"  aucune donnée via {name}")
            continue
        df = _frame(rows)
        df = df[(df["time"] >= pd.Timestamp(start_ms, unit="ms", tz="UTC"))
                & (df["time"] < pd.Timestamp(end_ms, unit="ms", tz="UTC"))].reset_index(drop=True)
        if df.empty:
            continue
        print(f"  {len(df)} bougies, du {df['time'].iloc[0]} au {df['time'].iloc[-1]}")
        if use_cache and end:
            os.makedirs(CACHE_DIR, exist_ok=True)
            df.to_csv(cache, index=False)
        return df, name
    raise RuntimeError(f"Impossible de récupérer l'historique de {symbol} {interval} (toutes les sources ont échoué).")


def load_csv(path):
    """Charge un CSV local (colonnes time, open, high, low, close, volume)."""
    df = pd.read_csv(path, parse_dates=["time"])
    if df["time"].dt.tz is None:
        df["time"] = df["time"].dt.tz_localize("UTC")
    return df.sort_values("time").reset_index(drop=True)
