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

function numCell(v, dec, extraClass = '') {
  if (v == null) return '<span class="num-value">—</span>';
  return `<span class="num-value${extraClass}">${v.toFixed(dec)}</span>`;
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

const N_BARS = 25;
const VB_W = 200, VB_H = 48;
const ZERO_Y = VB_H / 2, MAX_BAR = ZERO_Y - 3, SLOT_W = VB_W / N_BARS, BAR_W = SLOT_W - 2;

function buildTable(section) {
  const id = `${section.id}-body`;
  const colClass = { sts: 'col-sts', intraday: 'col-intraday', chg: 'col-chg', weekly: 'col-weekly', monthly: 'col-monthly', ytd: 'col-ytd', atrLow: 'col-atr-low', atrLowPrevFri: 'col-atr-wkchg', atrHigh: 'col-atr-high', atrHighPrevFri: 'col-atr-hchg' };
  const sortTh = (col, label) =>
    `<th class="${colClass[col]} sortable" data-sort="${col}" data-tbody="${id}">${label}<span class="sort-arrow"></span></th>`;
  return `<table class="ticker-table">
    <colgroup>
      <col class="col-market"><col class="col-index">
      <col class="col-rs"><col class="col-chart"><col class="col-sts">
      <col class="col-intraday"><col class="col-chg">
      <col class="col-weekly"><col class="col-monthly"><col class="col-ytd">
      <col class="col-atr-low"><col class="col-atr-wkchg">
      <col class="col-atr-high"><col class="col-atr-hchg">
    </colgroup>
    <thead><tr>
      <th class="col-market">${section.label}</th>
      <th class="col-index">Symbol</th>
      <th class="col-rs">1-Month RS</th>
      <th class="col-chart">1-Month Chart</th>
      ${sortTh('sts', 'RS_STS%')}
      ${sortTh('intraday', 'Intraday %')}
      ${sortTh('chg', 'Daily %')}
      ${sortTh('weekly', 'Weekly %')}
      ${sortTh('monthly', 'Monthly %')}
      ${sortTh('ytd', 'YTD %')}
      ${sortTh('atrLow', '52WL Dist (ATR%)')}
      ${sortTh('atrLowPrevFri', '52WL Dist (Prev Fri)')}
      ${sortTh('atrHigh', '52WH Dist (ATR%)')}
      ${sortTh('atrHighPrevFri', '52WH Dist (Prev Fri)')}
    </tr></thead>
    <tbody id="${id}"></tbody>
  </table>`;
}

function buildHistogramSVG(values, isSpy) {
  const maxAbs = isSpy ? 1 : Math.max(...values.filter(v => v !== null).map(Math.abs), 0.01);
  const maxVal = isSpy ? 0 : Math.max(...values.filter(v => v !== null));
  const maxIdx = isSpy ? -1 : values.indexOf(maxVal);

  const bars = values.map((v, i) => {
    if (v == null || isNaN(v)) return '';
    const isPos = v >= 0, isMax = i === maxIdx;
    const barH  = Math.abs(v) / maxAbs * MAX_BAR;
    const x = i * SLOT_W + 1;
    const y = isPos ? ZERO_Y - barH : ZERO_Y;
    const fill = isSpy  ? '#8c959f'
               : isMax  ? (isPos ? '#2da44e' : '#e5534b')
                        : (isPos ? '#1a7f37' : '#cf222e');
    return `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${BAR_W}" height="${Math.max(barH, 0.5).toFixed(1)}" fill="${fill}"><title>${v >= 0 ? '+' : ''}${v.toFixed(4)}</title></rect>`;
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
let _series = {}, _priceSeries = {}, _changes = {}, _weekly = {}, _monthly = {}, _ytd = {}, _intraday = {}, _atrMetrics = {};

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
    case 'atrLow':        return _atrMetrics?.[ticker]?.atr_low              ?? -Infinity;
    case 'atrLowPrevFri': return _atrMetrics?.[ticker]?.atr_low_prev_friday  ?? -Infinity;
    case 'atrHigh':       return _atrMetrics?.[ticker]?.atr_high             ?? -Infinity;
    case 'atrHighPrevFri':return _atrMetrics?.[ticker]?.atr_high_prev_friday ?? -Infinity;
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
  const maxAbsAtrLow  = maxAbs(order.map(({ ticker }) => _atrMetrics?.[ticker]?.atr_low  ?? null).filter(v => v != null));
  const maxAbsAtrHigh = maxAbs(order.map(({ ticker }) => _atrMetrics?.[ticker]?.atr_high ?? null).filter(v => v != null));

  document.getElementById(tbodyId).innerHTML = order.map(({ ticker, market }) => {
    const isSpy  = ticker === 'SPY';
    const values = (isSpy ? Array(N_BARS).fill(0) : (_series[ticker] ?? [])).slice(-N_BARS);

    let stsHtml = '<span class="num-value">—</span>';
    if (!isSpy && values.length) {
      const pr = percentRank(values, values[values.length - 1]);
      if (pr !== null) stsHtml = `<span class="num-value">${Math.round(pr * 100)}%</span>`;
    }

    const intra = _intraday[ticker] ?? null;
    const chg   = _changes[ticker]  ?? null;
    const wkly  = _weekly[ticker]   ?? null;
    const mnth  = _monthly[ticker]  ?? null;
    const ytdV  = _ytd[ticker]      ?? null;
    const m     = _atrMetrics?.[ticker];
    const atrLow         = m?.atr_low              ?? null;
    const atrLowPrevFri  = m?.atr_low_prev_friday  ?? null;
    const atrHigh        = m?.atr_high             ?? null;
    const atrHighPrevFri = m?.atr_high_prev_friday ?? null;

    return `<tr class="ticker-row">
      <td class="market-cell"><span class="ticker-symbol">${market}</span></td>
      <td class="index-cell">${ticker}</td>
      <td class="rs-cell">${buildHistogramSVG(values, isSpy)}</td>
      <td class="chart-cell">${buildLineChartSVG(_priceSeries[ticker])}</td>
      <td class="sts-cell">${stsHtml}</td>
      <td class="chg-cell" style="${intraBg(intra, maxAbsIntra)}"><span class="chg-text">${pctText(intra)}</span></td>
      <td class="chg-cell" style="${intraBg(chg, maxAbsChg)}"><span class="chg-text">${pctText(chg)}</span></td>
      <td class="chg-cell" style="${intraBg(wkly, maxAbsWeekly)}"><span class="chg-text">${pctText(wkly)}</span></td>
      <td class="chg-cell" style="${intraBg(mnth, maxAbsMonthly)}"><span class="chg-text">${pctText(mnth)}</span></td>
      <td class="chg-cell" style="${intraBg(ytdV, maxAbsYtd)}"><span class="chg-text">${pctText(ytdV)}</span></td>
      <td class="chg-cell" style="${intraBg(atrLow, maxAbsAtrLow)}"><span class="chg-text">${numCell(atrLow, 1)}</span></td>
      <td class="atr-cell prev-fri-cell">${numCell(atrLowPrevFri, 1)}</td>
      <td class="chg-cell" style="${intraBg(atrHigh, maxAbsAtrHigh)}"><span class="chg-text">${numCell(atrHigh, 1, atrHigh > 0 ? ' atr-new-high' : '')}</span></td>
      <td class="atr-cell prev-fri-cell">${numCell(atrHighPrevFri, 1)}</td>
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
    _priceSeries = latest.price_series    ?? {};
    _changes     = latest.daily_change    ?? {};
    _weekly      = latest.weekly_change   ?? {};
    _monthly     = latest.monthly_change  ?? {};
    _ytd         = latest.ytd_change      ?? {};
    _intraday    = latest.intraday_change ?? {};
    _atrMetrics  = latest.atr_metrics     ?? {};

    const TBODY_TO_ORDER = Object.fromEntries(
      config.sections.map(s => [`${s.id}-body`, s.rows.map(r => ({
        ticker: r.ticker,
        market: r.industry ?? r.label,
      }))]),
    );

    Object.entries(TBODY_TO_ORDER).forEach(([id, order]) => renderSection(order, id));
    updateSortArrows();

    document.querySelectorAll('th.sortable').forEach(th => {
      th.addEventListener('click', () => {
        const { tbody: tbodyId, sort: col } = th.dataset;
        const cur = sortState[tbodyId];
        sortState[tbodyId] = cur?.col === col ? { col, dir: -cur.dir } : { col, dir: -1 };
        renderSection(TBODY_TO_ORDER[tbodyId], tbodyId);
        updateSortArrows();
        syncBtns(TBODY_TO_ORDER);
      });
    });

    function applyGlobalSort(col, btn) {
      const isActive = btn.classList.contains('active');
      Object.entries(TBODY_TO_ORDER).forEach(([id, order]) => {
        if (isActive) delete sortState[id];
        else sortState[id] = { col, dir: -1 };
        renderSection(order, id);
      });
      updateSortArrows();
      syncBtns(TBODY_TO_ORDER);
    }

    function syncBtns(tbodyMap) {
      const ids = Object.keys(tbodyMap);
      btnWL.classList.toggle('active', ids.every(id => sortState[id]?.col === 'atrLowPrevFri'  && sortState[id]?.dir === -1));
      btnWH.classList.toggle('active', ids.every(id => sortState[id]?.col === 'atrHighPrevFri' && sortState[id]?.dir === -1));
    }

    const btnWL = document.getElementById('btn-wl-prev-fri');
    const btnWH = document.getElementById('btn-wh-prev-fri');
    btnWL.addEventListener('click', () => applyGlobalSort('atrLowPrevFri',  btnWL));
    btnWH.addEventListener('click', () => applyGlobalSort('atrHighPrevFri', btnWH));

  } catch (err) {
    showStatus('数据加载失败：' + err.message, true);
    console.error(err);
  }
})();
