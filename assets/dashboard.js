/* Shared dashboard logic. Each page sets window.DASHBOARD_CONFIG / DASHBOARD_DATA before loading. */
const CONFIG_PATH = window.DASHBOARD_CONFIG ?? './data/config.json';
const DATA_PATH   = window.DASHBOARD_DATA   ?? './data/latest.json';

function showStatus(msg, isError = false) {
  const el = document.getElementById('status');
  el.textContent = msg;
  el.style.display = 'block';
  el.classList.toggle('error', isError);
}


function toBeiJing(isoStr) {
  if (!isoStr) return '—';
  return new Date(isoStr).toLocaleString('zh-CN', {
    timeZone: 'Asia/Shanghai',
    year: 'numeric', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit',
  });
}

async function fetchJson(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} HTTP ${r.status}`);
  return r.json();
}

function pctText(v) {
  return v != null ? `${v >= 0 ? '+' : ''}${v.toFixed(2)}%` : '—';
}

function intraBg(val, maxAbs) {
  if (val == null) return '';
  const alpha = (0.08 + Math.min(Math.abs(val) / maxAbs, 1) * 0.42).toFixed(3);
  return val >= 0 ? `background:rgba(26,127,55,${alpha})` : `background:rgba(207,34,46,${alpha})`;
}

function percentRank(values, val) {
  const valid = values.filter(v => v !== null && !isNaN(v));
  if (valid.length <= 1) return null;
  return valid.filter(v => v < val).length / (valid.length - 1);
}

const N_BARS = 25, N_BARS_1W = 5;
const VB_W = 200, VB_H = 48;
const ZERO_Y = VB_H - 4, MAX_BAR = ZERO_Y - 3, SLOT_W = VB_W / N_BARS;
const BAR_GAP = 0.15, BAR_W = SLOT_W - BAR_GAP;

function buildTable(section) {
  const id = `${section.id}-body`;
  const colClass = { sts: 'col-sts', intraday: 'col-intraday', chg: 'col-chg', weekly: 'col-weekly', monthly: 'col-monthly', ytd: 'col-ytd', off52wl: 'col-atr-low', off52wlPrevFri: 'col-atr-wkchg' };
  const sortTh = (col, label) =>
    `<th class="${colClass[col]} sortable" data-sort="${col}" data-tbody="${id}">${label}<span class="sort-arrow"></span></th>`;
  return `<table class="ticker-table">
    <colgroup>
      <col class="col-index"><col class="col-market">
      <col class="col-rs-1w"><col class="col-rs"><col class="col-chart"><col class="col-sts">
      <col class="col-intraday"><col class="col-chg">
      <col class="col-weekly"><col class="col-monthly"><col class="col-ytd">
      <col class="col-atr-low"><col class="col-atr-wkchg">
    </colgroup>
    <thead><tr>
      <th class="col-index">Symbol</th>
      <th class="col-market">${section.label}</th>
      <th class="col-rs-1w">1-Week RS</th>
      <th class="col-rs">1-Month RS</th>
      <th class="col-chart">1-Month Chart</th>
      ${sortTh('sts', 'RS_STS%')}
      ${sortTh('intraday', 'Intraday %')}
      ${sortTh('chg', 'Daily %')}
      ${sortTh('weekly', 'Weekly %')}
      ${sortTh('monthly', 'Monthly %')}
      ${sortTh('ytd', 'YTD %')}
      ${sortTh('off52wl', 'Off 52WL %')}
      ${sortTh('off52wlPrevFri', 'Off 52WL (Prev Fri)')}
    </tr></thead>
    <tbody id="${id}"></tbody>
  </table>`;
}

function buildHistogramSVG(values, isSpy, nBars = N_BARS) {
  const valid = values.filter(v => v != null && !isNaN(v));
  if (!valid.length) return '';

  const minVal = isSpy ? 0 : Math.min(...valid);
  const maxVal = isSpy ? 0 : Math.max(...valid);
  const range  = maxVal - minVal || 0.001;

  const slotW = VB_W / nBars;
  const barW  = slotW - BAR_GAP;

  const bars = values.map((v, i) => {
    if (v == null || isNaN(v)) return '';
    const barH = isSpy ? 0 : ((v - minVal) / range) * MAX_BAR;
    const x    = (i * slotW + BAR_GAP / 2).toFixed(1);
    const y    = (ZERO_Y - barH).toFixed(1);
    const fill = isSpy ? '#8c959f' : (v === maxVal ? '#1a7f37' : '#93c5a8');
    return `<rect x="${x}" y="${y}" width="${barW.toFixed(2)}" height="${Math.max(barH, 0.5).toFixed(1)}" fill="${fill}"><title>${v.toFixed(4)}</title></rect>`;
  }).join('');

  return `<svg viewBox="0 0 ${VB_W} ${VB_H}" preserveAspectRatio="none">
    <line x1="0" y1="${ZERO_Y}" x2="${VB_W}" y2="${ZERO_Y}" stroke="#d0d7de" stroke-width="0.8"/>
    ${bars}</svg>`;
}

function buildLineChartSVG(values) {
  const pts = (values ?? []).filter(v => v != null);
  if (pts.length < 2) return '';
  const lo = Math.min(...pts), hi = Math.max(...pts);
  const range = hi - lo || 1;
  const W = 50, H = 22, pad = 2;
  const points = pts.map((v, i) => {
    const x = (pad + (i / (pts.length - 1)) * (W - 2 * pad)).toFixed(1);
    const y = (pad + (1 - (v - lo) / range) * (H - 2 * pad)).toFixed(1);
    return `${x},${y}`;
  }).join(' ');
  const color = pts[pts.length - 1] >= pts[0] ? '#1a7f37' : '#cf222e';
  return `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
    <polyline points="${points}" fill="none" stroke="${color}" stroke-width="1.2" stroke-linejoin="round" stroke-linecap="round"/>
  </svg>`;
}

const sortState = {};
let _series = {}, _series1w = {}, _priceSeries = {}, _changes = {}, _weekly = {}, _monthly = {}, _ytd = {}, _intraday = {}, _wlMetrics = {};

function getSortValue(ticker, col) {
  switch (col) {
    case 'sts': {
      const vals = (_series[ticker] ?? []).slice(-N_BARS);
      if (!vals.length) return -Infinity;
      return percentRank(vals, vals[vals.length - 1]) ?? -Infinity;
    }
    case 'intraday':      return _intraday[ticker]                           ?? -Infinity;
    case 'chg':           return _changes[ticker]                            ?? -Infinity;
    case 'weekly':        return _weekly[ticker]                             ?? -Infinity;
    case 'monthly':       return _monthly[ticker]                            ?? -Infinity;
    case 'ytd':           return _ytd[ticker]                                ?? -Infinity;
    case 'off52wl':        return _wlMetrics?.[ticker]?.off_52wl          ?? -Infinity;
    case 'off52wlPrevFri': return _wlMetrics?.[ticker]?.off_52wl_prev_fri ?? -Infinity;
    default: return 0;
  }
}

function updateSortArrows() {
  document.querySelectorAll('th.sortable').forEach(th => {
    const tbodyId = th.dataset.tbody;
    const col = th.dataset.sort;
    const ss = sortState[tbodyId];
    const arrow = th.querySelector('.sort-arrow');
    const active = ss?.col === col;
    th.classList.toggle('sort-active', active);
    arrow.textContent = active ? (ss.dir === -1 ? ' ▼' : ' ▲') : '';
  });
}

function renderSection(displayOrder, tbodyId) {
  const ss    = sortState[tbodyId];
  const order = ss
    ? [...displayOrder].sort((a, b) =>
        (getSortValue(a.ticker, ss.col) - getSortValue(b.ticker, ss.col)) * ss.dir)
    : displayOrder;

  const vals   = k => order.map(({ ticker }) => k[ticker]).filter(v => v != null);
  const maxAbs = arr => Math.max(...arr.map(Math.abs), 0.01);

  const maxAbsIntra   = maxAbs(vals(_intraday));
  const maxAbsChg     = maxAbs(vals(_changes));
  const maxAbsWeekly  = maxAbs(vals(_weekly));
  const maxAbsMonthly = maxAbs(vals(_monthly));
  const maxAbsYtd     = maxAbs(vals(_ytd));
  const maxAbsOff52wl = maxAbs(order.map(({ ticker }) => _wlMetrics?.[ticker]?.off_52wl ?? null).filter(v => v != null));

  document.getElementById(tbodyId).innerHTML = order.map(({ ticker, market }) => {
    const isSpy   = ticker === 'SPY';
    const values  = (isSpy ? Array(N_BARS).fill(0)    : (_series[ticker]   ?? [])).slice(-N_BARS);
    const values1w = (isSpy ? Array(N_BARS_1W).fill(0) : (_series1w[ticker] ?? [])).slice(-N_BARS_1W);

    let stsHtml = '<span class="num-value">—</span>';
    if (!isSpy && values.length) {
      const pr = percentRank(values, values[values.length - 1]);
      if (pr !== null) stsHtml = `<span class="num-value">${Math.round(pr * 100)}%</span>`;
    }

    const intra      = _intraday[ticker] ?? null;
    const chg        = _changes[ticker]  ?? null;
    const wkly       = _weekly[ticker]   ?? null;
    const mnth       = _monthly[ticker]  ?? null;
    const ytdV       = _ytd[ticker]      ?? null;
    const m          = _wlMetrics?.[ticker];
    const off52wl    = m?.off_52wl          ?? null;
    const off52wlFri = m?.off_52wl_prev_fri ?? null;

    return `<tr class="ticker-row">
      <td class="index-cell">${ticker}</td>
      <td class="market-cell"><span class="ticker-symbol">${market}</span></td>
      <td class="rs-cell">${buildHistogramSVG(values1w, isSpy, N_BARS_1W)}</td>
      <td class="rs-cell">${buildHistogramSVG(values, isSpy)}</td>
      <td class="chart-cell">${buildLineChartSVG(_priceSeries[ticker])}</td>
      <td class="sts-cell">${stsHtml}</td>
      <td class="chg-cell" style="${intraBg(intra, maxAbsIntra)}"><span class="chg-text">${pctText(intra)}</span></td>
      <td class="chg-cell" style="${intraBg(chg, maxAbsChg)}"><span class="chg-text">${pctText(chg)}</span></td>
      <td class="chg-cell" style="${intraBg(wkly, maxAbsWeekly)}"><span class="chg-text">${pctText(wkly)}</span></td>
      <td class="chg-cell" style="${intraBg(mnth, maxAbsMonthly)}"><span class="chg-text">${pctText(mnth)}</span></td>
      <td class="chg-cell" style="${intraBg(ytdV, maxAbsYtd)}"><span class="chg-text">${pctText(ytdV)}</span></td>
      <td class="chg-cell" style="${intraBg(off52wl, maxAbsOff52wl)}"><span class="chg-text">${pctText(off52wl)}</span></td>
      <td class="atr-cell prev-fri-cell">${off52wlFri != null ? off52wlFri.toFixed(2) : '—'}</td>
    </tr>`;
  }).join('');
}

(async () => {
  try {
    const [config, latest] = await Promise.all([
      fetchJson(CONFIG_PATH),
      fetchJson(DATA_PATH),
    ]);

    document.getElementById('dashboard').innerHTML = config.sections.map(buildTable).join('');

    if (!latest.date) {
      showStatus('数据尚未生成，等待首次 GitHub Actions 运行（北京时间每日 06:00）。');
      document.getElementById('dashboard').style.display = 'block';
      return;
    }

    document.getElementById('data-date').textContent = latest.date;
    document.getElementById('update-time').textContent = toBeiJing(latest.updated_at);
    document.getElementById('dashboard').style.display = 'block';

    _series      = latest.rs_series       ?? {};
    _series1w    = latest.rs_series_1w    ?? {};
    _priceSeries = latest.price_series    ?? {};
    _changes     = latest.daily_change    ?? {};
    _weekly      = latest.weekly_change   ?? {};
    _monthly     = latest.monthly_change  ?? {};
    _ytd         = latest.ytd_change      ?? {};
    _intraday    = latest.intraday_change ?? {};
    _wlMetrics   = latest.wl_metrics      ?? {};

    const TBODY_TO_ORDER = Object.fromEntries(
      config.sections.map(s => [`${s.id}-body`, s.rows.map(r => ({
        ticker: r.ticker,
        market: r.industry ?? r.label,
      }))]),
    );

    // Default sort: group tables by Off 52WL Prev Fri descending
    ['group-ew-body', 'group-body'].forEach(id => {
      if (id in TBODY_TO_ORDER) sortState[id] = { col: 'off52wlPrevFri', dir: -1 };
    });

    Object.entries(TBODY_TO_ORDER).forEach(([id, order]) => renderSection(order, id));
    updateSortArrows();

    document.querySelectorAll('th.sortable').forEach(th => {
      th.addEventListener('click', () => {
        const { tbody: tbodyId, sort: col } = th.dataset;
        const cur = sortState[tbodyId];
        sortState[tbodyId] = cur?.col === col ? { col, dir: -cur.dir } : { col, dir: -1 };
        renderSection(TBODY_TO_ORDER[tbodyId], tbodyId);
        updateSortArrows();
      });
    });

  } catch (err) {
    showStatus('数据加载失败：' + err.message, true);
    console.error(err);
  }
})();
