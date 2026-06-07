"""
Fetch daily OHLCV data via Yahoo Finance and compute RS (Price Ratio Relative Strength)
for all non-SPY tickers vs SPY.

RS line = ticker_close / SPY_close (raw ratio, last LOOKBACK trading days).
"""

import json
import logging
import os
import sys
from datetime import date, timedelta, datetime, UTC
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf
import exchange_calendars as xcals

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
BENCHMARK  = "SPY"
LOOKBACK   = 50
FETCH_DAYS = 400   # calendar days; covers 52W (≈365) + buffer

DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data")
LATEST_PATH = os.path.join(DATA_DIR, "latest.json")
CONFIG_PATH = os.path.join(DATA_DIR, "config.json")

# Extra config files: each maps to its own output JSON.
EXTRA_CONFIGS: list[tuple[str, str]] = [
    (os.path.join(DATA_DIR, "watchlist.json"),
     os.path.join(DATA_DIR, "watchlist_latest.json")),
]


def _tickers_from_config(path: str) -> list[str]:
    """Return ordered unique tickers from a config file."""
    with open(path) as f:
        cfg = json.load(f)
    seen, tickers = set(), []
    for section in cfg["sections"]:
        for row in section["rows"]:
            t = row["ticker"]
            if t not in seen:
                seen.add(t)
                tickers.append(t)
    return tickers


def load_config() -> list[str]:
    """Return all unique tickers across all config files."""
    tickers = _tickers_from_config(CONFIG_PATH)
    seen = set(tickers)
    for cfg_path, _ in EXTRA_CONFIGS:
        if os.path.exists(cfg_path):
            for t in _tickers_from_config(cfg_path):
                if t not in seen:
                    seen.add(t)
                    tickers.append(t)
    if BENCHMARK not in seen:
        tickers.insert(0, BENCHMARK)
    return tickers


TICKERS = load_config()


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


def get_last_week_friday(last_day: date) -> date:
    """Return the most recent Friday NYSE session strictly before last_day.

    If last_day is a Friday, returns the previous Friday (i.e. the week-start
    baseline is always the Friday of the prior week).
    Handles holidays by skipping non-session Fridays.
    """
    cal = xcals.get_calendar("XNYS")
    for i in range(1, 21):   # look back up to 3 weeks
        candidate = last_day - timedelta(days=i)
        if candidate.weekday() == 4 and cal.is_session(pd.Timestamp(candidate)):
            return candidate
    raise RuntimeError("Could not find last Friday trading session")


# ── Yahoo Finance ─────────────────────────────────────────────────────────────
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
        df = raw[["Open", "Close"]].copy()
        df.columns = ["adjOpen", "adjClose"]
        result[ticker] = df.dropna()
    else:
        for ticker in tickers:
            try:
                df = raw.xs(ticker, level=1, axis=1)[["Open", "Close"]].copy()
                df.columns = ["adjOpen", "adjClose"]
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


def compute_rs(data: dict) -> tuple[dict, dict]:
    """Return (rs_latest, rs_series).

    rs_series[ticker] = last LOOKBACK values of price-ratio RS (ticker/SPY).
    rs_latest[ticker] = the most recent RS ratio value.
    """
    spy_close = data[BENCHMARK]["adjClose"].dropna()
    results, series = {}, {"dates": []}

    for ticker in TICKERS:
        if ticker == BENCHMARK or ticker not in data:
            continue

        t_close = data[ticker]["adjClose"].dropna()
        t_aligned, spy_aligned = t_close.align(spy_close, join="inner")
        if t_aligned.empty:
            log.warning("  No overlapping dates for %s – skipping", ticker)
            continue

        rs_ratio = (t_aligned / spy_aligned).dropna().iloc[-LOOKBACK:]
        if len(rs_ratio) < 2:
            continue

        if not series["dates"]:
            series["dates"] = [d.strftime("%Y-%m-%d") for d in rs_ratio.index]

        series[ticker]  = [round(float(v), 4) for v in rs_ratio.values]
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


def compute_rs_series_weekly(rs_series: dict, n: int = 5) -> dict:
    """Return last n values of rs_series for the 1-week RS histogram."""
    out: dict = {"dates": rs_series.get("dates", [])[-n:]}
    for ticker, vals in rs_series.items():
        if ticker == "dates":
            continue
        out[ticker] = vals[-n:]
    return out


def compute_price_series(data: dict, n: int = 21) -> dict:
    """Return {ticker: [last n adjClose prices]} for the 1-month line chart."""
    out = {}
    for ticker in TICKERS:
        if ticker not in data:
            continue
        close = data[ticker]["adjClose"].dropna()
        if len(close) >= 2:
            out[ticker] = [round(float(v), 4) for v in close.iloc[-n:].values]
    return out


def _changes_since(data: dict, cutoff: pd.Timestamp) -> dict:
    """Return {ticker: pct_change} — latest Close vs last available Close at or before cutoff."""
    out = {}
    for ticker in TICKERS:
        if ticker not in data:
            continue
        close = data[ticker]["adjClose"].dropna()
        base_s = close[close.index <= cutoff]
        if base_s.empty:
            continue
        base = float(base_s.iloc[-1])
        if base:
            out[ticker] = round((close.iloc[-1] - base) / base * 100, 2)
    return out


def compute_weekly_changes(data: dict, last_day: date) -> dict:
    return _changes_since(data, pd.Timestamp(get_last_week_friday(last_day)))


def compute_monthly_changes(data: dict, last_day: date) -> dict:
    return _changes_since(data, pd.Timestamp(last_day.replace(day=1) - timedelta(days=1)))


def compute_ytd_changes(data: dict, last_day: date) -> dict:
    return _changes_since(data, pd.Timestamp(date(last_day.year - 1, 12, 31)))


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


def compute_52wl_metrics(data: dict, last_day: date) -> dict:
    """Return {ticker: {off_52wl, off_52wl_prev_fri}} — % above 52W low."""
    prev_ts = pd.Timestamp(get_last_week_friday(last_day))
    log.info("52WL baseline Friday: %s", prev_ts.date())

    metrics = {}
    for ticker in TICKERS:
        if ticker not in data:
            continue
        close = data[ticker]["adjClose"].dropna()
        if close.empty:
            continue

        current = float(close.iloc[-1])
        low_now = float(close.iloc[-252:].min())
        if low_now == 0:
            continue

        off_now = round((current - low_now) / low_now * 100, 2)

        # Prev Friday snapshot
        off_fri = None
        close_to_fri = close[close.index <= prev_ts]
        if not close_to_fri.empty:
            fri_close = float(close_to_fri.iloc[-1])
            fri_low   = float(close_to_fri.iloc[-252:].min())
            if fri_low:
                off_fri = round((fri_close - fri_low) / fri_low * 100, 2)

        metrics[ticker] = {"off_52wl": off_now, "off_52wl_prev_fri": off_fri}
    return metrics


# ── Persistence ───────────────────────────────────────────────────────────────
def _filter_metrics(metrics: dict, subset: set[str]) -> dict:
    """Return metrics filtered to the given ticker subset."""
    out = {}
    for k, v in metrics.items():
        if not isinstance(v, dict):
            out[k] = v
        elif k == "rs_series":
            out[k] = {"dates": v.get("dates", [])}
            out[k].update({t: v[t] for t in subset if t in v})
        else:
            out[k] = {t: v[t] for t in subset if t in v}
    return out


def save_data(trade_date: date, metrics: dict, output_path: str = LATEST_PATH,
              subset: set[str] | None = None) -> None:
    if subset is not None:
        metrics = _filter_metrics(metrics, subset)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    payload = {
        "date":       trade_date.isoformat(),
        "updated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "params":     {"lookback": LOOKBACK},
        **metrics,
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    log.info("Saved → %s  (date=%s)", output_path, payload["date"])


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

    data = fetch_yahoo(TICKERS, start, last_day)

    rs_latest, rs_series = compute_rs(data)
    log.info("RS latest: %s", {k: round(v, 4) for k, v in rs_latest.items()})
    metrics = {
        "rs":              rs_latest,
        "rs_series":       rs_series,
        "rs_series_1w":    compute_rs_series_weekly(rs_series),
        "price_series":    compute_price_series(data),
        "daily_change":    compute_daily_changes(data),
        "weekly_change":   compute_weekly_changes(data, last_day),
        "monthly_change":  compute_monthly_changes(data, last_day),
        "ytd_change":      compute_ytd_changes(data, last_day),
        "intraday_change": compute_intraday_changes(data),
        "wl_metrics":      compute_52wl_metrics(data, last_day),
    }
    save_data(last_day, metrics, LATEST_PATH,
              subset=set(_tickers_from_config(CONFIG_PATH)))
    for cfg_path, out_path in EXTRA_CONFIGS:
        if os.path.exists(cfg_path):
            save_data(last_day, metrics, out_path,
                      subset=set(_tickers_from_config(cfg_path)))


if __name__ == "__main__":
    main()