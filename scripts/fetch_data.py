"""
Fetch daily OHLCV data and compute VARS for all non-SPY tickers vs SPY.
Data source is configured via data/config.json: "data_source": "yahoo" | "tiingo"

VARS formula (mattishenner / Jeff Sun):
  vol_adj_change_t = (Close_t - Close_{t-1}) / ATR_t
  VARS = sum(vol_adj_change_asset, N) - sum(vol_adj_change_SPY, N)
  ATR uses Wilder's smoothing: ewm(com = period-1, adjust=False)
"""

import json
import logging
import os
import subprocess
import sys
import time
from datetime import date, timedelta, datetime, UTC
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf
import exchange_calendars as xcals

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
TIINGO_TOKEN      = os.environ.get("TIINGO_TOKEN", "")
BENCHMARK         = "SPY"
ATR_PERIOD        = 14
LOOKBACK          = 50
FETCH_DAYS        = 400   # calendar days; covers 52W (≈365) + ATR warm-up buffer
INTER_REQUEST_SEC = 1     # delay between requests within a batch
BATCH_SIZE        = 49    # Tiingo free tier: 50 requests/hour; use 49 to be safe
BATCH_PAUSE_SEC   = 3660  # 61 min — wait for the hourly window to reset

DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data")
LATEST_PATH = os.path.join(DATA_DIR, "latest.json")
CONFIG_PATH = os.path.join(DATA_DIR, "config.json")
CACHE_PATH  = os.path.join(DATA_DIR, "fetch_cache.json")


def load_config() -> tuple[list[str], str]:
    """Return (tickers, data_source) from config.json."""
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    data_source = cfg.get("data_source", "tiingo")
    seen, tickers = set(), []
    for section in cfg["sections"]:
        for row in section["rows"]:
            t = row["ticker"]
            if t not in seen:
                seen.add(t)
                tickers.append(t)
    if BENCHMARK not in seen:
        tickers.insert(0, BENCHMARK)
    return tickers, data_source


TICKERS, DATA_SOURCE = load_config()


# ── Trading calendar ──────────────────────────────────────────────────────────
def get_last_trading_day() -> date:
    """Return the most recent completed NYSE session."""
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
class RateLimitError(Exception):
    """Raised when Tiingo returns 429; signals the caller to wait and retry."""


def fetch_tiingo(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Return a DataFrame with columns [adjOpen, adjHigh, adjLow, adjClose].

    Retries up to 3 times on transient network errors.
    Raises RateLimitError on 429 so the caller can pause and retry.
    Raises ValueError if the ticker genuinely has no data (caller should skip).
    """
    if not TIINGO_TOKEN:
        raise EnvironmentError("TIINGO_TOKEN is not set")

    raw = []
    for attempt in range(3):
        try:
            resp = requests.get(
                f"https://api.tiingo.com/tiingo/daily/{ticker}/prices",
                params={"startDate": start.isoformat(), "endDate": end.isoformat(),
                        "token": TIINGO_TOKEN, "resampleFreq": "daily"},
                timeout=30,
            )
        except requests.exceptions.RequestException as e:
            if attempt < 2:
                log.warning("  Network error for %s (attempt %d): %s – retrying", ticker, attempt + 1, e)
                time.sleep(5 * (attempt + 1))
                continue
            raise

        if resp.status_code == 429:
            raise RateLimitError(f"429 rate limit hit on {ticker}")

        resp.raise_for_status()
        raw = resp.json()
        break

    if not raw:
        raise ValueError(f"No data returned for {ticker}")

    df = pd.DataFrame(raw)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize().dt.tz_localize(None)
    df = df.set_index("date").sort_index()

    needed = {"adjOpen", "adjHigh", "adjLow", "adjClose"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{ticker} missing columns: {missing}")

    return df[["adjOpen", "adjHigh", "adjLow", "adjClose"]]


# ── Yahoo Finance (batch) ─────────────────────────────────────────────────────
def fetch_yahoo(tickers: list[str], start: date, end: date) -> dict[str, pd.DataFrame]:
    """Batch-download all tickers in one yfinance call. No API key required."""
    end_excl = end + timedelta(days=1)
    log.info("Yahoo Finance: downloading %d tickers %s → %s …", len(tickers), start, end)

    raw = yf.download(
        tickers,
        start=start.isoformat(),
        end=end_excl.isoformat(),
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    raw.index = raw.index.tz_localize(None) if raw.index.tz else raw.index
    raw.index = raw.index.normalize()

    result = {}
    if len(tickers) == 1:
        ticker = tickers[0]
        df = raw[["Open", "High", "Low", "Close"]].copy()
        df.columns = ["adjOpen", "adjHigh", "adjLow", "adjClose"]
        result[ticker] = df.dropna()
    else:
        for ticker in tickers:
            try:
                df = raw.xs(ticker, level=1, axis=1)[["Open", "High", "Low", "Close"]].copy()
                df.columns = ["adjOpen", "adjHigh", "adjLow", "adjClose"]
                df = df.dropna()
                if df.empty:
                    log.warning("  No data for %s", ticker)
                else:
                    result[ticker] = df
            except KeyError:
                log.warning("  %s not found in Yahoo response", ticker)

    missing = [t for t in tickers if t not in result]
    if missing:
        log.warning("Missing tickers: %s", missing)
    return result


def load_cache(trade_date: date) -> dict[str, pd.DataFrame]:
    """Load per-ticker DataFrames from today's cache file, if it exists."""
    if not os.path.exists(CACHE_PATH):
        return {}
    with open(CACHE_PATH) as f:
        raw = json.load(f)
    if raw.get("date") != trade_date.isoformat():
        return {}   # stale cache from a previous day
    result = {}
    for ticker, records in raw.get("tickers", {}).items():
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        result[ticker] = df
    log.info("Cache hit: %d tickers already fetched for %s", len(result), trade_date)
    return result


def save_cache(trade_date: date, data: dict[str, pd.DataFrame]) -> None:
    """Persist fetched DataFrames to cache so re-runs skip already-done tickers."""
    serialisable = {}
    for ticker, df in data.items():
        records = df.reset_index()
        records["date"] = records["date"].dt.strftime("%Y-%m-%d")
        serialisable[ticker] = records.to_dict(orient="records")
    with open(CACHE_PATH, "w") as f:
        json.dump({"date": trade_date.isoformat(), "tickers": serialisable}, f)


def git_push_cache() -> None:
    """Commit and push the cache file so a re-run can resume from this point."""
    try:
        diff = subprocess.run(["git", "diff", "--quiet", CACHE_PATH], capture_output=True)
        if diff.returncode == 0:
            return   # no changes
        subprocess.run(["git", "add", CACHE_PATH], check=True)
        subprocess.run(
            ["git", "commit", "-m", f"cache: interim save ({datetime.now(UTC).strftime('%H:%M UTC')})"],
            check=True,
        )
        subprocess.run(["git", "pull", "--rebase"], check=True)
        subprocess.run(["git", "push"], check=True)
        log.info("  Cache committed and pushed.")
    except subprocess.CalledProcessError as e:
        log.warning("  git push cache failed (non-fatal, workflow will retry at end): %s", e)


def fetch_all(tickers: list[str], start: date, end: date,
              trade_date: date) -> dict[str, pd.DataFrame]:
    """Fetch all tickers via the configured data source."""
    log.info("Data source: %s", DATA_SOURCE)

    if DATA_SOURCE == "yahoo":
        return fetch_yahoo(tickers, start, end)

    # ── Tiingo: resume from cache, batch with pauses ──────────────────────────
    data = load_cache(trade_date)
    remaining = [t for t in tickers if t not in data]

    if not remaining:
        log.info("All tickers already cached – skipping fetch.")
        return data

    log.info("Fetching %d/%d tickers (batch size %d, pause %ds) …",
             len(remaining), len(tickers), BATCH_SIZE, BATCH_PAUSE_SEC)

    req_count = 0
    try:
        for ticker in remaining:
            if req_count > 0 and req_count % BATCH_SIZE == 0:
                git_push_cache()
                log.info("  [%d requests done] Pausing %ds …", req_count, BATCH_PAUSE_SEC)
                time.sleep(BATCH_PAUSE_SEC)
            log.info("  [%d/%d] %s", req_count + 1, len(remaining), ticker)
            for rate_retry in range(5):
                try:
                    data[ticker] = fetch_tiingo(ticker, start, end)
                    save_cache(trade_date, data)
                    break
                except RateLimitError:
                    wait = BATCH_PAUSE_SEC * (rate_retry + 1)
                    log.warning("  Rate limit hit – pausing %ds before retrying %s …", wait, ticker)
                    git_push_cache()
                    time.sleep(wait)
                except Exception as e:
                    log.warning("  Skipping %s – %s", ticker, e)
                    break
            req_count += 1
            time.sleep(INTER_REQUEST_SEC)
    except Exception:
        log.warning("Fetch interrupted at %d requests – pushing cache before exit", req_count)
        git_push_cache()
        raise

    return data


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
      results = {ticker: float}
      series  = {"dates": [...], ticker: [float × LOOKBACK], ...}
    """
    spy_cum = (data[BENCHMARK]["adjClose"].diff() / wilder_atr(data[BENCHMARK], ATR_PERIOD)
               ).rolling(LOOKBACK).sum()

    results, series = {}, {"dates": []}

    for ticker in TICKERS:
        if ticker == BENCHMARK or ticker not in data:
            continue
        df = data[ticker]
        cum = (df["adjClose"].diff() / wilder_atr(df, ATR_PERIOD)).rolling(LOOKBACK).sum()

        t_aligned, spy_aligned = cum.align(spy_cum, join="inner")
        if t_aligned.empty:
            log.warning("  No overlapping dates for %s – skipping", ticker)
            continue

        recent = (t_aligned - spy_aligned).dropna().iloc[-LOOKBACK:]
        if recent.empty:
            log.warning("  No valid VARS data for %s – skipping", ticker)
            continue

        if not series["dates"]:
            series["dates"] = [d.strftime("%Y-%m-%d") for d in recent.index]

        series[ticker]  = [round(float(v), 4) for v in recent.values]
        results[ticker] = series[ticker][-1]

    return results, series


def compute_daily_changes(data: dict) -> dict:
    """Return {ticker: daily_pct_change} — Close vs previous Close."""
    changes = {}
    for ticker in TICKERS:
        if ticker not in data:
            continue
        close = data[ticker]["adjClose"].dropna()
        if len(close) >= 2:
            changes[ticker] = round(
                (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100, 2
            )
    return changes


def compute_weekly_changes(data: dict) -> dict:
    """Return {ticker: weekly_pct_change} — Close vs Close 5 trading days ago."""
    changes = {}
    for ticker in TICKERS:
        if ticker not in data:
            continue
        close = data[ticker]["adjClose"].dropna()
        if len(close) >= 6:
            changes[ticker] = round(
                (close.iloc[-1] - close.iloc[-6]) / close.iloc[-6] * 100, 2
            )
    return changes


def compute_intraday_changes(data: dict) -> dict:
    """Return {ticker: intraday_pct_change} — latest Close vs latest Open."""
    changes = {}
    for ticker in TICKERS:
        if ticker not in data:
            continue
        df = data[ticker].dropna(subset=["adjOpen", "adjClose"])
        if df.empty:
            continue
        open_ = float(df["adjOpen"].iloc[-1])
        close = float(df["adjClose"].iloc[-1])
        if open_ != 0:
            changes[ticker] = round((close - open_) / open_ * 100, 2)
    return changes


def compute_atr_metrics(data: dict) -> dict:
    """
    For each ticker compute ATR%-normalised distance from 52W extremes:
      atr_low  = ((Close - Low_52w)  / Low_52w)  / (ATR / Close)
      atr_high = ((Close - High_52w) / High_52w) / (ATR / Close)  — positive = new 52W high
    """
    metrics = {}
    for ticker in TICKERS:
        if ticker not in data:
            continue
        df    = data[ticker]
        close = df["adjClose"]
        atr   = wilder_atr(df, ATR_PERIOD)

        current = float(close.iloc[-1])
        atr_val = float(atr.iloc[-1])
        if atr_val == 0 or current == 0:
            continue

        window   = close.iloc[-252:]
        low_52w  = float(window.min())
        high_52w = float(window.max())
        atr_pct  = atr_val / current

        metrics[ticker] = {
            "atr_low":  round((current - low_52w)  / low_52w  / atr_pct, 2),
            "atr_high": round((current - high_52w) / high_52w / atr_pct, 2),
        }
    return metrics


# ── Persistence ───────────────────────────────────────────────────────────────
def save_data(trade_date: date, vars_result: dict, vars_series: dict,
              daily_changes: dict, weekly_changes: dict,
              intraday_changes: dict, atr_metrics: dict) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    date_str = trade_date.isoformat()

    payload = {
        "date":             date_str,
        "updated_at":       datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "params":           {"atr_period": ATR_PERIOD, "lookback": LOOKBACK},
        "vars":             vars_result,
        "vars_series":      vars_series,
        "daily_change":     daily_changes,
        "weekly_change":    weekly_changes,
        "intraday_change":  intraday_changes,
        "atr_metrics":      atr_metrics,
    }

    with open(LATEST_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    log.info("Saved  date=%s  vars=%s", date_str, payload["vars"])


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    last_day = get_last_trading_day()
    et_now   = datetime.now(ZoneInfo("America/New_York"))
    log.info("ET now: %s  →  last trading day: %s", et_now.strftime("%Y-%m-%d %H:%M"), last_day)

    if os.path.exists(LATEST_PATH):
        with open(LATEST_PATH) as f:
            saved_date = json.load(f).get("date")
        if saved_date == last_day.isoformat():
            log.info("Already have data for %s – nothing to do.", last_day)
            sys.exit(0)
        log.info("Saved date is %s, need %s – fetching.", saved_date, last_day)

    start = last_day - timedelta(days=FETCH_DAYS)

    data = fetch_all(TICKERS, start, last_day, last_day)

    vars_result, vars_series = compute_vars(data)
    daily_changes    = compute_daily_changes(data)
    weekly_changes   = compute_weekly_changes(data)
    intraday_changes = compute_intraday_changes(data)
    atr_metrics      = compute_atr_metrics(data)
    log.info("VARS result: %s", {k: round(v, 4) for k, v in vars_result.items()})
    save_data(last_day, vars_result, vars_series, daily_changes, weekly_changes, intraday_changes, atr_metrics)


if __name__ == "__main__":
    main()
