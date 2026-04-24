"""
Fetch daily OHLCV data from Tiingo, compute VARS for all non-SPY tickers vs SPY,
and save results to data/ for GitHub Pages to serve.

VARS formula (mattishenner / Jeff Sun):
  vol_adj_change_t = (Close_t - Close_{t-1}) / ATR_t
  VARS = sum(vol_adj_change_asset, N) - sum(vol_adj_change_SPY, N)
  ATR uses Wilder's smoothing: ewm(com = period-1, adjust=False)
"""

import json
import logging
import os
import sys
from datetime import date, timedelta, datetime, timezone
from zoneinfo import ZoneInfo

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
TICKERS      = ["SPY", "RSP", "QQQ", "QQQE", "IWM", "DIA",
                "IJS", "IJT", "IJJ", "IJK", "IVE", "IVW",
                "XLRE", "XLU", "XLV", "XLF", "XLP", "XLB",
                "XLE", "XLI", "XLY", "XLC", "XLK"]
ATR_PERIOD   = 14
LOOKBACK     = 50
FETCH_DAYS   = 400   # calendar days; covers 52W (≈365) + ATR warm-up buffer

DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data")
LATEST_PATH = os.path.join(DATA_DIR, "latest.json")


# ── Trading calendar ──────────────────────────────────────────────────────────
def get_last_trading_day() -> date:
    """Return the most recent completed NYSE session.

    Uses ET time: if it's before 4:15 PM ET, today's session isn't closed yet,
    so fall back to the previous trading day.
    """
    cal = xcals.get_calendar("XNYS")
    et_now  = datetime.now(ZoneInfo("America/New_York"))
    cutoff  = et_now.replace(hour=16, minute=15, second=0, microsecond=0)
    ref     = et_now.date() if et_now >= cutoff else et_now.date() - timedelta(days=1)
    for i in range(10):
        candidate = ref - timedelta(days=i)
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


def compute_atr_metrics(data: dict) -> dict:
    """
    For each ticker compute:
      atr_low  = ATR% multiple from 52W low
               = ((Close - Low_52w) / Low_52w) / (ATR / Close)
      atr_high = ATR% multiple from 52W high  (negative = new 52W high)
               = ((High_52w - Close) / High_52w) / (ATR / Close)
    """
    metrics = {}
    for ticker in TICKERS:
        df    = data[ticker]
        close = df["adjClose"]
        atr   = wilder_atr(df, ATR_PERIOD)

        current  = float(close.iloc[-1])
        atr_val  = float(atr.iloc[-1])
        if atr_val == 0 or current == 0:
            continue

        window   = close.iloc[-252:]          # up to 252 trading days ≈ 52 weeks
        low_52w  = float(window.min())
        high_52w = float(window.max())
        atr_pct  = atr_val / current          # ATR as fraction of price

        metrics[ticker] = {
            "atr_low":  round((current - low_52w)  / low_52w  / atr_pct, 2),
            "atr_high": round((current - high_52w) / high_52w / atr_pct, 2),
        }
    return metrics


def compute_daily_changes(data: dict) -> dict:
    """Return {ticker: daily_pct_change} for all tickers including benchmark."""
    changes = {}
    for ticker in TICKERS:
        close = data[ticker]["adjClose"].dropna()
        if len(close) >= 2:
            changes[ticker] = round(
                (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100, 2
            )
    return changes


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
def save_data(trade_date: date, vars_result: dict, vars_series: dict,
              daily_changes: dict, atr_metrics: dict) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    date_str = trade_date.isoformat()

    payload = {
        "date":         date_str,
        "updated_at":   datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "params":       {"atr_period": ATR_PERIOD, "lookback": LOOKBACK},
        "vars":         vars_result,
        "vars_series":  vars_series,
        "daily_change": daily_changes,
        "atr_metrics":  atr_metrics,
    }

    with open(LATEST_PATH, "w") as f:
        json.dump(payload, f, indent=2)

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
    daily_changes = compute_daily_changes(data)
    atr_metrics   = compute_atr_metrics(data)
    log.info("VARS result: %s", {k: round(v, 4) for k, v in vars_result.items()})
    save_data(last_day, vars_result, vars_series, daily_changes, atr_metrics)


if __name__ == "__main__":
    main()
