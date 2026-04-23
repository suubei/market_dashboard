"""
Fetch daily OHLCV data from Tiingo, compute VARS for RSP/QQQ/QQQE/IWM/DIA vs SPY,
and save results to data/ for GitHub Pages to serve.

VARS formula (mattishenner / Jeff Sun):
  vol_adj_change_t = (Close_t - Close_{t-1}) / ATR_t
  VARS = sum(vol_adj_change_asset, N) - sum(vol_adj_change_SPY, N)
  ATR uses Wilder's smoothing: ewm(com = period-1, adjust=False)
"""

import csv
import json
import logging
import os
import sys
from datetime import date, timedelta, datetime

import pandas as pd
import requests
import exchange_calendars as xcals

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
TIINGO_TOKEN = os.environ.get("TIINGO_TOKEN", "")
BENCHMARK    = "SPY"
TICKERS      = ["SPY", "RSP", "QQQ", "QQQE", "IWM", "DIA"]
ATR_PERIOD   = 14
LOOKBACK     = 25
FETCH_DAYS   = 130   # calendar days; enough for ATR warm-up + lookback window

DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data")
HISTORY_DIR = os.path.join(DATA_DIR, "history")
CSV_PATH    = os.path.join(DATA_DIR, "vars_history.csv")
LATEST_PATH = os.path.join(DATA_DIR, "latest.json")


# ── Trading calendar ──────────────────────────────────────────────────────────
def get_last_trading_day() -> date:
    """Return the most recent completed NYSE session (UTC date context)."""
    cal = xcals.get_calendar("XNYS")
    today_utc = date.today()
    for i in range(10):
        candidate = today_utc - timedelta(days=i)
        if cal.is_session(pd.Timestamp(candidate)):
            return candidate
    raise RuntimeError("Could not find last trading day within 10 days")


# ── Tiingo API ────────────────────────────────────────────────────────────────
def fetch_tiingo(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Return a DataFrame with columns [adjHigh, adjLow, adjClose]."""
    if not TIINGO_TOKEN:
        raise EnvironmentError("TIINGO_TOKEN is not set")

    resp = requests.get(
        f"https://api.tiingo.com/tiingo/daily/{ticker}/prices",
        params={"startDate": start.isoformat(), "endDate": end.isoformat(),
                "token": TIINGO_TOKEN, "resampleFreq": "daily"},
        timeout=30,
    )
    resp.raise_for_status()
    raw = resp.json()
    if not raw:
        raise ValueError(f"Tiingo returned no data for {ticker} ({start} – {end})")

    df = pd.DataFrame(raw)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize().dt.tz_localize(None)
    df = df.set_index("date").sort_index()

    needed = {"adjHigh", "adjLow", "adjClose"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Tiingo response for {ticker} missing columns: {missing}")

    return df[["adjHigh", "adjLow", "adjClose"]]


# ── VARS calculation ──────────────────────────────────────────────────────────
def wilder_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Wilder's smoothed ATR (ewm with alpha = 1/period)."""
    high, low, close = df["adjHigh"], df["adjLow"], df["adjClose"]
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def compute_vars(data: dict) -> tuple:
    """
    Returns (results, series):
      results = {ticker: float}              – latest VARS value per ticker
      series  = {"dates": [...], ticker: [float × LOOKBACK], ...}
    """
    spy_cum = (data[BENCHMARK]["adjClose"].diff() / wilder_atr(data[BENCHMARK], ATR_PERIOD)
               ).rolling(LOOKBACK).sum()

    results, series = {}, {"dates": []}

    for ticker in TICKERS:
        if ticker == BENCHMARK:
            continue
        df = data[ticker]
        cum = (df["adjClose"].diff() / wilder_atr(df, ATR_PERIOD)).rolling(LOOKBACK).sum()

        t_aligned, spy_aligned = cum.align(spy_cum, join="inner")
        if t_aligned.empty:
            raise ValueError(f"No overlapping dates between {ticker} and {BENCHMARK}")

        recent = (t_aligned - spy_aligned).dropna().iloc[-LOOKBACK:]

        if not series["dates"]:
            series["dates"] = [d.strftime("%Y-%m-%d") for d in recent.index]

        series[ticker]  = [round(float(v), 4) for v in recent.values]
        results[ticker] = series[ticker][-1]

    return results, series


# ── Persistence ───────────────────────────────────────────────────────────────
def load_history() -> list:
    """Read vars_history.csv; returns list of row dicts."""
    if not os.path.exists(CSV_PATH):
        return []
    with open(CSV_PATH, newline="") as f:
        return list(csv.DictReader(f))


def save_data(trade_date: date, vars_result: dict, vars_series: dict) -> None:
    os.makedirs(HISTORY_DIR, exist_ok=True)
    date_str = trade_date.isoformat()

    payload = {
        "date":        date_str,
        "updated_at":  datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "params":      {"atr_period": ATR_PERIOD, "lookback": LOOKBACK},
        "vars":        {k: round(v, 4) for k, v in vars_result.items()},
        "vars_series": vars_series,
    }

    # Per-day JSON snapshot
    with open(os.path.join(HISTORY_DIR, f"{date_str}.json"), "w") as f:
        json.dump(payload, f, indent=2)

    # latest.json – what the dashboard reads
    with open(LATEST_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    # Append to CSV (skip if date already present)
    existing_dates = {row["date"] for row in load_history()}
    csv_is_new = not os.path.exists(CSV_PATH)
    if date_str not in existing_dates:
        with open(CSV_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            if csv_is_new:
                writer.writerow(["date", "ticker", "vars"])
            for ticker, value in payload["vars"].items():
                writer.writerow([date_str, ticker, value])

    log.info("Saved  date=%s  vars=%s", date_str, payload["vars"])


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    last_day = get_last_trading_day()
    log.info("Last trading day: %s", last_day)

    if os.path.exists(LATEST_PATH):
        with open(LATEST_PATH) as f:
            if json.load(f).get("date") == last_day.isoformat():
                log.info("Data already up to date – nothing to do.")
                sys.exit(0)

    start = last_day - timedelta(days=FETCH_DAYS)
    log.info("Fetching %d tickers from %s to %s", len(TICKERS), start, last_day)

    data = {}
    for ticker in TICKERS:
        log.info("  Fetching %s …", ticker)
        data[ticker] = fetch_tiingo(ticker, start, last_day)

    vars_result, vars_series = compute_vars(data)
    log.info("VARS result: %s", {k: round(v, 4) for k, v in vars_result.items()})
    save_data(last_day, vars_result, vars_series)


if __name__ == "__main__":
    main()
