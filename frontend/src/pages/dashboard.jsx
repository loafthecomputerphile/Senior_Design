import React, { useEffect, useMemo, useState } from "react";

/* ------------------------------------------------------------------ *
 *  Mock data. Replace `SEED` with records read from IndexedDB (SFR14).
 *  Shape of one record:
 *    id, ts (ms), risk ("Low" | "Medium" | "High"),
 *    result ("Negative" | "Uncertain" | "Positive"), confidence (0-100),
 *    votes [3 x class], signal {neg, unc, pos},
 *    snr (dB), clipping (string), duration (s), gate (bool)
 * ------------------------------------------------------------------ */

const DAY = 86400000;
const NOW = Date.now();

const rec = (id, daysAgo, risk, result, confidence, votes, signal, snr, duration) => ({
  id,
  ts: NOW - daysAgo * DAY - id * 977000,
  risk,
  result,
  confidence,
  votes,
  signal,
  snr,
  clipping: "None",
  duration,
  gate: true,
});

const SEED = [
  rec(1, 1, "Medium", "Positive", 62, ["Positive", "Positive", "Negative"], { neg: 0.31, unc: 0.18, pos: 0.62 }, 21, 9.4),
  rec(2, 3, "Low", "Negative", 81, ["Negative", "Negative", "Negative"], { neg: 0.81, unc: 0.1, pos: 0.12 }, 24, 8.7),
  rec(3, 5, "Low", "Negative", 74, ["Negative", "Negative", "Uncertain"], { neg: 0.74, unc: 0.2, pos: 0.09 }, 19, 10.2),
  rec(4, 8, "Low", "Negative", 69, ["Negative", "Negative", "Negative"], { neg: 0.69, unc: 0.16, pos: 0.14 }, 22, 9.1),
  rec(5, 12, "Medium", "Uncertain", 55, ["Uncertain", "Uncertain", "Positive"], { neg: 0.22, unc: 0.55, pos: 0.3 }, 18, 11.3),
  rec(6, 17, "Low", "Negative", 77, ["Negative", "Negative", "Negative"], { neg: 0.77, unc: 0.11, pos: 0.13 }, 23, 8.9),
  rec(7, 21, "High", "Positive", 71, ["Positive", "Positive", "Positive"], { neg: 0.15, unc: 0.12, pos: 0.71 }, 20, 9.8),
  rec(8, 26, "Medium", "Positive", 58, ["Positive", "Negative", "Positive"], { neg: 0.34, unc: 0.2, pos: 0.58 }, 17, 10.6),
  rec(9, 33, "Low", "Negative", 83, ["Negative", "Negative", "Negative"], { neg: 0.83, unc: 0.09, pos: 0.1 }, 25, 8.4),
  rec(10, 41, "Low", "Negative", 72, ["Negative", "Uncertain", "Negative"], { neg: 0.72, unc: 0.25, pos: 0.12 }, 21, 9.5),
  rec(11, 58, "Medium", "Positive", 61, ["Positive", "Positive", "Uncertain"], { neg: 0.2, unc: 0.3, pos: 0.61 }, 19, 10.0),
  rec(12, 74, "Low", "Negative", 79, ["Negative", "Negative", "Negative"], { neg: 0.79, unc: 0.12, pos: 0.11 }, 22, 9.2),
];

/* ---- constants ---------------------------------------------------- */

const RISK = {
  Low: {
    cls: "low",
    advice: "Keep an eye on how you feel. Test again if symptoms start or a known exposure happens.",
  },
  Medium: {
    cls: "med",
    advice: "Consider taking a COVID-19 test, and wear a mask around other people until you know more.",
  },
  High: {
    cls: "high",
    advice: "Take an official test soon. If you have symptoms, contact a healthcare provider.",
  },
};

const SHORT = { Negative: "Neg", Uncertain: "Unc", Positive: "Pos" };
const RANGES = ["7d", "30d", "90d", "All"];
const RANGE_DAYS = { "7d": 7, "30d": 30, "90d": 90, All: null };
const DISCLAIMER = "Pre-screening tool only, not a diagnosis.";

/* ---- helpers ------------------------------------------------------ */

const fmtDate = (ts) =>
  new Date(ts).toLocaleString("en-US", { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });

const fmtDay = (ts) => new Date(ts).toLocaleDateString("en-US", { month: "short", day: "numeric" });

function voteSummary(r) {
  const count = r.votes.filter((v) => v === r.result).length;
  return { count, short: SHORT[r.result] };
}

function download(name, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}

function exportCsv(records) {
  const head = ["date", "risk", "result", "confidence_pct", "votes", "neg", "unc", "pos", "snr_db", "clipping", "duration_s"];
  const rows = records.map((r) =>
    [new Date(r.ts).toISOString(), r.risk, r.result, r.confidence, r.votes.join("|"), r.signal.neg, r.signal.unc, r.signal.pos, r.snr, r.clipping, r.duration].join(",")
  );
  download("cough-history.csv", [head.join(","), ...rows].join("\n"), "text/csv");
}

function exportJson(records) {
  download("cough-history.json", JSON.stringify(records, null, 2), "application/json");
}

/* Icons */

const Svg = ({ children, size = 20 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    {children}
  </svg>
);

const I = {
  grid: () => (
    <Svg>
      <rect x="4" y="4" width="6.5" height="6.5" rx="1.8" />
      <rect x="13.5" y="4" width="6.5" height="6.5" rx="1.8" />
      <rect x="4" y="13.5" width="6.5" height="6.5" rx="1.8" />
      <rect x="13.5" y="13.5" width="6.5" height="6.5" rx="1.8" />
    </Svg>
  ),
  mic: () => (
    <Svg>
      <rect x="9" y="3" width="6" height="11" rx="3" />
      <path d="M5.5 11.5a6.5 6.5 0 0 0 13 0M12 18v3" />
    </Svg>
  ),
  help: () => (
    <Svg>
      <circle cx="12" cy="12" r="8.5" />
      <path d="M9.6 9.5a2.5 2.5 0 1 1 3.6 2.2c-.8.4-1.2 1-1.2 1.8M12 16.8v.1" />
    </Svg>
  ),
  text: () => (
    <Svg>
      <path d="M4 18 9 6l5 12M5.8 14h6.4M16 18l2.5-6 2.5 6M17 16h3" />
    </Svg>
  ),
  contrast: () => (
    <Svg>
      <circle cx="12" cy="12" r="8.5" />
      <path d="M12 3.5v17a8.5 8.5 0 0 0 0-17Z" fill="currentColor" stroke="none" />
    </Svg>
  ),
  lung: () => (
    <Svg>
      <path d="M12 4v7M12 11c-1.5 1-2.5 1.5-3.5 1.5M12 11c1.5 1 2.5 1.5 3.5 1.5" />
      <path d="M8.5 8C6 9 4 14 4 17.5 4 19 5 20 6.3 20c1.6 0 2.7-1 2.7-2.6V8.3ZM15.5 8C18 9 20 14 20 17.5c0 1.5-1 2.5-2.3 2.5-1.6 0-2.7-1-2.7-2.6V8.3Z" />
    </Svg>
  ),
  check: () => (
    <Svg size={14}>
      <circle cx="12" cy="12" r="8.5" />
      <path d="m8.5 12.3 2.4 2.4 4.6-5" />
    </Svg>
  ),
  warn: () => (
    <Svg size={14}>
      <path d="M12 4 3.5 19h17Z" />
      <path d="M12 10v4M12 16.6v.1" />
    </Svg>
  ),
  alert: () => (
    <Svg size={14}>
      <path d="M8.2 3.5h7.6l4.7 4.7v7.6l-4.7 4.7H8.2l-4.7-4.7V8.2Z" />
      <path d="M12 8v4.5M12 15.6v.1" />
    </Svg>
  ),
  info: () => (
    <Svg size={16}>
      <circle cx="12" cy="12" r="8.5" />
      <path d="M12 11v5M12 7.9v.1" />
    </Svg>
  ),
  download: () => (
    <Svg size={16}>
      <path d="M12 4v11M7.5 10.5 12 15l4.5-4.5M5 19.5h14" />
    </Svg>
  ),
  trash: () => (
    <Svg size={16}>
      <path d="M5 7h14M10 7V4.5h4V7M7 7l.8 12.5h8.4L17 7M10.3 11v5M13.7 11v5" />
    </Svg>
  ),
  close: () => (
    <Svg size={18}>
      <path d="m6.5 6.5 11 11M17.5 6.5l-11 11" />
    </Svg>
  ),
  chevron: () => (
    <Svg size={16}>
      <path d="m9.5 6 6 6-6 6" />
    </Svg>
  ),
  logout: () => (
    <Svg>
      <path d="M14 4h4.5A1.5 1.5 0 0 1 20 5.5v13a1.5 1.5 0 0 1-1.5 1.5H14M10 8l-4 4 4 4M6 12h9" />
    </Svg>
  ),
};

const RISK_ICON = { Low: I.check, Medium: I.warn, High: I.alert };

/* ---- small pieces -------------------------------------------------- */

function RiskBadge({ risk }) {
  const Icon = RISK_ICON[risk];
  return (
    <span className={`risk ${RISK[risk].cls}`}>
      <Icon />
      {risk} risk
    </span>
  );
}

function Seg({ options, value, onChange, label }) {
  return (
    <div className="seg" role="group" aria-label={label}>
      {options.map((o) => (
        <button key={o} type="button" className={o === value ? "on" : ""} aria-pressed={o === value} onClick={() => onChange(o)}>
          {o}
        </button>
      ))}
    </div>
  );
}

function Modal({ title, onClose, children }) {
  useEffect(() => {
    const onKey = (e) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="scrim" onMouseDown={onClose}>
      <div className="modal" role="dialog" aria-modal="true" aria-label={title} onMouseDown={(e) => e.stopPropagation()}>
        <button type="button" className="iconBtn modalClose" aria-label="Close" onClick={onClose}>
          <I.close />
        </button>
        {children}
      </div>
    </div>
  );
}

/* ---- trend chart: thin line, one highlighted point ----------------- */

function TrendChart({ records, range }) {
  const [hover, setHover] = useState(null);

  const W = 600, H = 230, L = 34, R = 14, T = 38, B = 28;
  const pts = useMemo(() => [...records].sort((a, b) => a.ts - b.ts), [records]);

  if (pts.length < 2) {
    return <div className="chartEmpty">Save at least 2 results in this range to see a trend.</div>;
  }

  const days = RANGE_DAYS[range];
  const start = days ? NOW - days * DAY : pts[0].ts;
  const span = Math.max(NOW - start, 1);
  const x = (ts) => L + ((ts - start) / span) * (W - L - R);
  const y = (c) => T + (1 - (c - 40) / 60) * (H - T - B);

  const xy = pts.map((p) => [x(p.ts), y(p.confidence)]);
  const line = xy.reduce((d, [px, py], i) => {
    if (i === 0) return `M${px},${py}`;
    const [qx, qy] = xy[i - 1];
    const mx = (qx + px) / 2;
    return `${d} C${mx},${qy} ${mx},${py} ${px},${py}`;
  }, "");
  const area = `${line} L${xy[xy.length - 1][0]},${H - B} L${xy[0][0]},${H - B} Z`;

  const active = hover ?? pts.length - 1;
  const [ax, ay] = xy[active];
  const ap = pts[active];
  const label = `${ap.confidence}% · ${ap.risk}`;
  const bw = 112;
  const bx = Math.min(Math.max(ax - bw / 2, L), W - R - bw);

  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Confidence over time">
      <defs>
        <linearGradient id="areaFill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#5B7BFF" stopOpacity=".28" />
          <stop offset="100%" stopColor="#5B7BFF" stopOpacity="0" />
        </linearGradient>
      </defs>

      {[100, 80, 65, 50].map((v) => (
        <g key={v}>
          <line x1={L} x2={W - R} y1={y(v)} y2={y(v)} stroke="rgba(20,27,61,.14)" strokeWidth="1" strokeDasharray={v === 65 || v === 50 ? "4 5" : "0"} />
          <text x={L - 8} y={y(v) + 3.5} textAnchor="end" className="axis">{v}</text>
        </g>
      ))}

      <path d={area} fill="url(#areaFill)" />
      <path d={line} fill="none" stroke="#5B7BFF" strokeWidth="2" strokeLinecap="round" />

      <line x1={ax} x2={ax} y1={ay} y2={H - B} stroke="#5B7BFF" strokeWidth="1" strokeDasharray="3 4" opacity=".6" pointerEvents="none" />

{xy.map(([px, py], i) =>
  i !== active ? (
    <circle key={pts[i].id} cx={px} cy={py} r="2.5" fill="#fff" stroke="#5B7BFF" strokeWidth="1.2" opacity=".85" pointerEvents="none" />
  ) : null
)}

<circle cx={ax} cy={ay} r="11" fill="#5B7BFF" opacity=".16" pointerEvents="none" />
<circle cx={ax} cy={ay} r="5" fill="#fff" stroke="#5B7BFF" strokeWidth="2.4" pointerEvents="none" />

<g pointerEvents="none">
  <rect x={bx} y={ay - 38} width={bw} height="24" rx="12" fill="#141B3D" />
  <text x={bx + bw / 2} y={ay - 22} textAnchor="middle" className="tip">{label}</text>
</g>

{/* hover targets go last so they sit on top of everything */}
{xy.map(([px, py], i) => (
  <circle
    key={`hit-${pts[i].id}`}
    cx={px}
    cy={py}
    r="13"
    fill="transparent"
    tabIndex={0}
    aria-label={`${fmtDay(pts[i].ts)}: ${pts[i].confidence}% ${pts[i].risk} risk`}
    onMouseEnter={() => setHover(i)}
    onMouseLeave={() => setHover(null)}
    onFocus={() => setHover(i)}
    onBlur={() => setHover(null)}
  />
))}

      <text x={L} y={H - 6} className="axis">{fmtDay(start)}</text>
      <text x={W - R} y={H - 6} textAnchor="end" className="axis">Today</text>
    </svg>
  );
}

/* ---- main component ------------------------------------------------ */

export default function Dashboard({ onStartTest, user, onSignOut }) {
  const [records, setRecords] = useState(SEED);
  const [range, setRange] = useState("30d");
  const [filter, setFilter] = useState([]);
  const [sort, setSort] = useState("Newest");
  const [selectedId, setSelectedId] = useState(null);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [showHelp, setShowHelp] = useState(false);
  const [big, setBig] = useState(false);
  const [contrast, setContrast] = useState(false);
  const [online, setOnline] = useState(typeof navigator === "undefined" ? true : navigator.onLine);

  useEffect(() => {
    const up = () => setOnline(true);
    const down = () => setOnline(false);
    window.addEventListener("online", up);
    window.addEventListener("offline", down);
    return () => {
      window.removeEventListener("online", up);
      window.removeEventListener("offline", down);
    };
  }, []);

  const start = () => {
    if (!online) return;
    if (onStartTest) onStartTest();
    else window.location.hash = "#/recorder";
  };

  const sorted = useMemo(() => [...records].sort((a, b) => b.ts - a.ts), [records]);
  const latest = sorted[0];

  const inRange = useMemo(() => {
    const d = RANGE_DAYS[range];
    return d ? sorted.filter((r) => r.ts >= NOW - d * DAY) : sorted;
  }, [sorted, range]);

  const list = useMemo(() => {
    const f = filter.length ? sorted.filter((r) => filter.includes(r.risk)) : sorted;
    return sort === "Newest" ? f : [...f].reverse();
  }, [sorted, filter, sort]);

  const counts = useMemo(() => {
    const c = { Low: 0, Medium: 0, High: 0 };
    records.forEach((r) => (c[r.risk] += 1));
    return c;
  }, [records]);

  const week = records.filter((r) => r.ts >= NOW - 7 * DAY).length;
  const maxCount = Math.max(1, ...Object.values(counts));
  const selected = records.find((r) => r.id === selectedId);

  const toggleFilter = (risk) => setFilter((f) => (f.includes(risk) ? f.filter((x) => x !== risk) : [...f, risk]));
  const removeOne = (id) => {
    setRecords((rs) => rs.filter((r) => r.id !== id));
    setSelectedId(null);
  };

  return (
    <>
      <style>{CSS}</style>

      <div className={`page${big ? " big" : ""}${contrast ? " contrast" : ""}`}>
        <div className="frame">
          {/* ---------- slim icon sidebar ---------- */}
          <aside className="rail" aria-label="Navigation">
            <div className="logo" aria-hidden="true"><I.lung /></div>
            <nav className="railNav">
              <button type="button" className="iconBtn active" aria-label="Dashboard" aria-current="page" title="Dashboard"><I.grid /></button>
              <button type="button" className="iconBtn" aria-label="Recorder" title="Recorder" onClick={start} disabled={!online}><I.mic /></button>
              <button type="button" className="iconBtn" aria-label="How it works" title="How it works" onClick={() => setShowHelp(true)}><I.help /></button>
            </nav>
            <div className="railEnd">
  <button type="button" className={`iconBtn${big ? " active" : ""}`} aria-label="Larger text" aria-pressed={big} title="Larger text" onClick={() => setBig(!big)}><I.text /></button>

  <button type="button" className={`iconBtn${contrast ? " active" : ""}`} aria-label="High contrast" aria-pressed={contrast} title="High contrast" onClick={() => setContrast(!contrast)}><I.contrast /></button>

  <button type="button" className="iconBtn" aria-label="Sign out" title="Sign out" onClick={onSignOut}><I.logout /></button>
</div>
          </aside>

          {/* ---------- main ---------- */}
          <main className="main">
            <header className="head">
              <div>
                <h1>Your cough history</h1>
                <p className="sub">
  {user?.name ? `${user.name} · ` : ""}
  {latest ? `Last test ${fmtDate(latest.ts)}` : "No tests yet"}
</p>
              </div>
              <button type="button" className="btnDark" onClick={start} disabled={!online}>
                <I.mic /> Start cough test
              </button>
            </header>

            <div className="strip" role="note">
              <I.info /> {DISCLAIMER} Your results stay on this device, and the server stores nothing.
            </div>

            {!online && (
              <div className="strip offline" role="status">
                You're offline. History and trends still work, but a new test needs a connection.
              </div>
            )}

            {records.length === 0 ? (
              <section className="card welcome">
                <h2>Take your first test</h2>
                <p className="muted">Three coughs, under a minute. Your results are saved here, on this device only.</p>
                <ol className="steps">
                  <li>Allow microphone access.</li>
                  <li>Give three loud, clear coughs, one at a time.</li>
                  <li>Read your result and save it to track changes.</li>
                </ol>
                <button type="button" className="btnDark" onClick={start} disabled={!online}><I.mic /> Start cough test</button>
              </section>
            ) : (
              <div className="bento">
                {/* Latest result: vivid hero */}
                <section className="card hero" aria-labelledby="h-latest">
                  <div className="heroTop">
                    <span id="h-latest" className="label onBlue">Latest result</span>
                    <RiskBadge risk={latest.risk} />
                  </div>
                  <div className="heroNum">
                    {latest.confidence}<span className="pct">%</span>
                  </div>
                  <p className="heroSub">confidence · {fmtDate(latest.ts)}</p>

                  <div className="heroPanel">
                    <div className="gauge" role="img" aria-label={`Confidence ${latest.confidence} percent`}>
                      <div className="gaugeFill" style={{ width: `${latest.confidence}%` }} />
                      <span className="mark" style={{ left: "50%" }} />
                      <span className="mark" style={{ left: "65%" }} />
                    </div>
                    <div className="gaugeKey">
                      <span style={{ left: "50%" }}>50 gate</span>
                      <span style={{ left: "65%" }}>65 high</span>
                    </div>
                    <p className="advice">{RISK[latest.risk].advice}</p>
                    <button type="button" className="btnGlass" onClick={() => setSelectedId(latest.id)}>View details</button>
                  </div>
                </section>

                {/* Trend */}
                <section className="card trend" aria-labelledby="h-trend">
                  <div className="cardHead">
                    <h2 id="h-trend" className="label">Confidence over time</h2>
                    <Seg options={RANGES} value={range} onChange={setRange} label="Time range" />
                  </div>
                  <TrendChart records={inRange} range={range} />
                  <p className="legend">Dashed lines mark the 50% result gate and the 65% high-risk level.</p>
                </section>

                {/* Vote breakdown */}
                <section className="card votes" aria-labelledby="h-votes">
                  <h2 id="h-votes" className="label">Vote breakdown</h2>
                  <div className="chips">
                    {latest.votes.map((v, i) => (
                      <div className="chip" key={i}>
                        <span className="chipN">Cough {i + 1}</span>
                        <span className="chipV">{SHORT[v]}</span>
                      </div>
                    ))}
                  </div>
                  <p className="voteLine">
                    {voteSummary(latest).count === 1
                      ? `Split vote, decided by signal strength: ${latest.result}`
                      : `${voteSummary(latest).count} of 3 → ${latest.result}`}
                  </p>
                </section>

                {/* Recording quality */}
                <section className="card quality" aria-labelledby="h-quality">
                  <h2 id="h-quality" className="label">Recording quality</h2>
                  <dl className="rows">
                    <div><dt>Signal-to-noise</dt><dd>{latest.snr} dB</dd></div>
                    <div><dt>Clipping</dt><dd>{latest.clipping}</dd></div>
                    <div><dt>Duration</dt><dd>{latest.duration} s</dd></div>
                    <div><dt>Cough check</dt><dd><span className="tag">{latest.gate ? "Passed" : "Failed"}</span></dd></div>
                  </dl>
                </section>

                {/* History list */}
                <section className="card history" aria-labelledby="h-history">
                  <div className="cardHead">
                    <h2 id="h-history" className="label">All results</h2>
                    <Seg options={["Newest", "Oldest"]} value={sort} onChange={setSort} label="Sort order" />
                  </div>
                  <div className="filters" role="group" aria-label="Filter by risk">
                    {["Low", "Medium", "High"].map((r) => (
                      <button key={r} type="button" className={`pill${filter.includes(r) ? " on" : ""}`} aria-pressed={filter.includes(r)} onClick={() => toggleFilter(r)}>
                        {r}
                      </button>
                    ))}
                  </div>
                  <ul className="list">
                    {list.length === 0 && <li className="none">No results match this filter.</li>}
                    {list.map((r) => {
                      const v = voteSummary(r);
                      return (
                        <li key={r.id}>
                          <button type="button" className="row" onClick={() => setSelectedId(r.id)}>
                            <span className="rowDate">{fmtDate(r.ts)}</span>
                            <RiskBadge risk={r.risk} />
                            <span className="rowConf">{r.confidence}%</span>
                            <span className="rowVote">{v.count}/3 {v.short}</span>
                            <I.chevron />
                          </button>
                        </li>
                      );
                    })}
                  </ul>
                </section>

                {/* Activity summary */}
                <section className="card activity" aria-labelledby="h-activity">
                  <h2 id="h-activity" className="label">Activity</h2>
                  <div className="actNums">
                    <div><strong>{records.length}</strong><span>total tests</span></div>
                    <div><strong>{week}</strong><span>this week</span></div>
                  </div>
                  <div className="bars">
                    {["Low", "Medium", "High"].map((r) => (
                      <div className="barCol" key={r}>
                        <span className="barN">{counts[r]}</span>
                        <div className="bar" style={{ height: `${18 + (counts[r] / maxCount) * 82}px` }} />
                        <span className="barL">{r}</span>
                      </div>
                    ))}
                  </div>
                </section>

                {/* Data controls */}
                <section className="card data" aria-labelledby="h-data">
                  <h2 id="h-data" className="label">Your data</h2>
                  <div className="dataBtns">
                    <button type="button" className="btnGlass" onClick={() => exportCsv(sorted)}><I.download /> Export CSV</button>
                    <button type="button" className="btnGlass" onClick={() => exportJson(sorted)}><I.download /> Export JSON</button>
                    <button type="button" className="btnGlass danger" onClick={() => setConfirmDelete(true)}><I.trash /> Delete all</button>
                  </div>
                  <p className="legend">Export a file to share with a clinician.</p>
                </section>
              </div>
            )}
          </main>
        </div>

        {/* mobile sticky start button */}
        <div className="stickyStart">
          <button type="button" className="btnDark" onClick={start} disabled={!online}><I.mic /> Start cough test</button>
        </div>
      </div>

      {/* ---------- dialogs ---------- */}
      {selected && (
        <Modal title="Result details" onClose={() => setSelectedId(null)}>
          <div className="mHead">
            <RiskBadge risk={selected.risk} />
            <span className="muted">{fmtDate(selected.ts)}</span>
          </div>
          <div className="mNum">{selected.confidence}<span className="pct">%</span></div>
          <p className="muted">confidence · {voteSummary(selected).count} of 3 → {selected.result}</p>

          <div className="chips">
            {selected.votes.map((v, i) => (
              <div className="chip" key={i}>
                <span className="chipN">Cough {i + 1}</span>
                <span className="chipV">{SHORT[v]}</span>
              </div>
            ))}
          </div>

          <dl className="rows">
            <div><dt>Signal strength</dt><dd>Neg {selected.signal.neg.toFixed(2)} / Unc {selected.signal.unc.toFixed(2)} / Pos {selected.signal.pos.toFixed(2)}</dd></div>
            <div><dt>Quality</dt><dd>{selected.snr} dB · clipping {selected.clipping.toLowerCase()} · {selected.duration} s</dd></div>
          </dl>

          <p className="advice dark">{RISK[selected.risk].advice}</p>
          <p className="legend">{DISCLAIMER}</p>

          <div className="mBtns">
            <button type="button" className="btnGlass danger" onClick={() => removeOne(selected.id)}><I.trash /> Delete this result</button>
            <button type="button" className="btnDark" onClick={() => { setSelectedId(null); start(); }} disabled={!online}>Test again</button>
          </div>
        </Modal>
      )}

      {confirmDelete && (
        <Modal title="Delete all results" onClose={() => setConfirmDelete(false)}>
          <h2 className="mTitle">Delete all results?</h2>
          <p className="muted">This removes every saved result from this device. It can't be undone.</p>
          <div className="mBtns">
            <button type="button" className="btnGlass" onClick={() => setConfirmDelete(false)}>Keep results</button>
            <button type="button" className="btnDark" onClick={() => { setRecords([]); setConfirmDelete(false); }}>Delete all</button>
          </div>
        </Modal>
      )}

      {showHelp && (
        <Modal title="How it works" onClose={() => setShowHelp(false)}>
          <h2 className="mTitle">How it works</h2>
          <ol className="steps">
            <li>Press Start cough test and allow the microphone.</li>
            <li>Give three loud, clear coughs, one at a time.</li>
            <li>We check the audio, classify each cough, and combine the three into one risk level.</li>
            <li>Save the result to track how you're doing over time.</li>
          </ol>
          <p className="legend">{DISCLAIMER} Confirm any concern with an official test.</p>
        </Modal>
      )}
    </>
  );
}

/* ---- styles -------------------------------------------------------- */

const CSS = `
@import url("https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;500;600;700&display=swap");

.page, .page * {
  box-sizing: border-box;
}

.page {
  --navy: #141B3D;
  --muted: #4F5882;
  --accent: #5B7BFF;

  /* Glass colors */
  --glass: rgba(255, 255, 255, 0.38);
  --glass-line: rgba(255, 255, 255, 0.55);

  --shadow: 0 10px 40px rgba(90, 100, 180, 0.15);

  min-height: 100vh;
  padding: 24px;

  background: #E6EAF3;

  font-family:
    Manrope,
    Inter,
    ui-sans-serif,
    -apple-system,
    "Segoe UI",
    sans-serif;

  font-size: 15px;
  color: var(--navy);
  -webkit-font-smoothing: antialiased;
}

.page.big {
  font-size: 17.5px;
}

.page.contrast {
  --muted: #262E57;
  --glass: rgba(255, 255, 255, 0.62);
  --glass-line: rgba(255, 255, 255, 0.9);
}

.page button {
  font: inherit;
  color: inherit;
  cursor: pointer;
}

.page button:disabled {
  cursor: not-allowed;
  opacity: 0.45;
}

.page circle:focus,
.page circle:focus-visible {
  outline: none;
}

.page :where(h1, h2, p, dl, dd, ol, ul) {
  margin: 0;
  padding: 0;
}

/* =========================================================
   MAIN GRADIENT CONTAINER
   ========================================================= */

.frame {
  display: flex;
  gap: 28px;
  min-height: calc(100vh - 48px);
  padding: 28px;
  border-radius: 40px;

  /*
    Layer 1: soft blue glow
    Layer 2: soft purple glow
    Layer 3: overall blue/lavender gradient
  */
  background:
    radial-gradient(
      circle at 15% 20%,
      rgba(120, 145, 255, 0.75),
      transparent 40%
    ),
    radial-gradient(
      circle at 85% 30%,
      rgba(170, 155, 255, 0.65),
      transparent 45%
    ),
    linear-gradient(
      135deg,
      #AEBBE5 0%,
      #C4CCE9 45%,
      #D9DDEB 100%
    );

  box-shadow:
    0 30px 80px rgba(90, 100, 180, 0.22);
}

/* =========================================================
   SIDEBAR
   ========================================================= */

.rail {
  width: 64px;
  flex-shrink: 0;

  display: flex;
  flex-direction: column;
  align-items: center;

  padding: 18px 0;
  border-radius: 32px;

  background: var(--glass);
  border: 1px solid var(--glass-line);

  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);

  box-shadow: var(--shadow);
}

.logo {
  width: 44px;
  height: 44px;
  border-radius: 50%;

  display: grid;
  place-items: center;

  color: #fff;

  background:
    linear-gradient(
      135deg,
      #5B7BFF,
      #7B6CF6
    );

  box-shadow:
    0 8px 20px rgba(91, 123, 255, 0.35);
}

.railNav {
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin-top: 36px;
}

.railEnd {
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin-top: auto;
}

.iconBtn {
  width: 44px;
  height: 44px;
  border-radius: 50%;

  display: grid;
  place-items: center;

  border: 1px solid transparent;
  background: transparent;
  color: var(--muted);

  transition:
    background 0.2s,
    color 0.2s;
}

.iconBtn:hover:not(:disabled) {
  background: rgba(255, 255, 255, 0.5);
  color: var(--navy);
}

.iconBtn.active {
  background: #fff;
  color: var(--navy);

  box-shadow:
    0 6px 16px rgba(90, 100, 180, 0.2);
}

/* =========================================================
   MAIN
   ========================================================= */

.main {
  flex: 1;
  min-width: 0;

  display: flex;
  flex-direction: column;
  gap: 18px;
}

.head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  flex-wrap: wrap;

  padding: 4px 4px 0;
}

.head h1 {
  font-size: 2.1em;
  font-weight: 300;
  letter-spacing: -0.035em;
  line-height: 1.1;
}

.sub {
  margin-top: 6px;
  font-size: 0.9em;
  color: var(--muted);
}

/* =========================================================
   INFO STRIP
   ========================================================= */

.strip {
  display: flex;
  align-items: center;
  gap: 8px;

  align-self: flex-start;

  padding: 9px 16px;
  border-radius: 999px;

  font-size: 0.82em;
  color: var(--muted);

  background: var(--glass);
  border: 1px solid var(--glass-line);

  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
}

.strip svg {
  flex-shrink: 0;
}

.strip.offline {
  color: var(--navy);
  background: rgba(255, 255, 255, 0.7);
}

/* =========================================================
   BUTTONS
   ========================================================= */

.btnDark {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;

  height: 46px;
  padding: 0 24px;

  border: 0;
  border-radius: 999px;

  background: #0A0A0F;
  color: #fff !important;

  font-size: 0.95em;
  font-weight: 600;

  box-shadow:
    0 10px 24px rgba(10, 10, 15, 0.25);

  transition: transform 0.15s;
}

.btnDark:hover:not(:disabled) {
  transform: translateY(-1px);
}

.btnGlass {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;

  height: 40px;
  padding: 0 18px;

  border-radius: 999px;
  border: 1px solid rgba(255, 255, 255, 0.7);

  background: rgba(255, 255, 255, 0.55);

  font-size: 0.86em;
  font-weight: 600;

  transition: background 0.2s;
}

.btnGlass:hover:not(:disabled) {
  background: rgba(255, 255, 255, 0.85);
}

/* =========================================================
   BENTO GRID
   ========================================================= */

.bento {
  display: grid;
  grid-template-columns: repeat(12, minmax(0, 1fr));
  gap: 20px;
}

/* =========================================================
   GLASS CARDS
   ========================================================= */

.card {
  position: relative;

  padding: 26px;
  border-radius: 24px;

  background: var(--glass);
  border: 1px solid var(--glass-line);

  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);

  box-shadow: var(--shadow);

  min-width: 0;
}

.label {
  font-size: 0.85em;
  font-weight: 600;
  color: var(--muted);
}

.label.onBlue {
  color: rgba(255, 255, 255, 0.85);
}

.muted {
  color: var(--muted);
  font-size: 0.92em;
}

.cardHead {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
}

/* =========================================================
   HERO GRADIENT
   ========================================================= */

.hero {
  grid-column: span 5;
  grid-row: span 2;

  display: flex;
  flex-direction: column;

  border-color: rgba(255, 255, 255, 0.55);

  /*
    Soft white glow over blue/purple gradient
  */
  background:
    radial-gradient(
      circle at 20% 10%,
      rgba(255, 255, 255, 0.25),
      transparent 35%
    ),
    linear-gradient(
      145deg,
      #627FFF 0%,
      #796CF4 35%,
      #A5B5FF 65%,
      #DCE4FF 100%
    );

  color: #fff;
}

.heroTop {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  flex-wrap: wrap;
}

.heroNum {
  margin-top: 22px;

  font-size: 5.6em;
  font-weight: 300;
  letter-spacing: -0.05em;
  line-height: 1;

  font-variant-numeric: tabular-nums;
}

.pct {
  font-size: 0.4em;
  margin-left: 4px;
  font-weight: 400;
  opacity: 0.8;
  letter-spacing: 0;
}

.heroSub {
  margin-top: 8px;
  font-size: 0.9em;
  color: rgba(255, 255, 255, 0.88);
}

/* =========================================================
   HERO INNER GLASS PANEL
   ========================================================= */

.heroPanel {
  margin-top: auto;
  padding: 22px;

  border-radius: 20px;

  color: var(--navy);

  background: rgba(255, 255, 255, 0.55);
  border: 1px solid rgba(255, 255, 255, 0.8);

  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
}

.gauge {
  position: relative;

  height: 14px;
  border-radius: 999px;

  background: rgba(91, 123, 255, 0.16);
}

.gaugeFill {
  height: 100%;
  border-radius: 999px;

  background:
    linear-gradient(
      90deg,
      #7B6CF6,
      #5B7BFF
    );

  box-shadow:
    inset 0 2px 3px rgba(255, 255, 255, 0.5),
    0 4px 10px rgba(91, 123, 255, 0.35);
}

.mark {
  position: absolute;
  top: -4px;
  bottom: -4px;

  width: 2px;

  background: var(--navy);
  opacity: 0.5;

  border-radius: 2px;
}

.gaugeKey {
  position: relative;

  height: 20px;
  margin-top: 8px;

  font-size: 0.74em;
  color: var(--muted);
}

.gaugeKey span {
  position: absolute;

  transform: translateX(-50%);

  white-space: nowrap;
}

.advice {
  margin: 12px 0 16px;

  font-size: 0.92em;
  line-height: 1.55;
}

.advice.dark {
  margin: 18px 0 8px;
}

/* =========================================================
   TREND
   ========================================================= */

.trend {
  grid-column: span 7;
}

.chart {
  display: block;

  width: 100%;
  height: auto;

  margin-top: 8px;
  overflow: visible;
}

.chart .axis {
  font-size: 11px;
  fill: #4F5882;
  font-family: inherit;
}

.chart .tip {
  font-size: 12px;
  font-weight: 600;
  fill: #fff;
  font-family: inherit;
}

.chartEmpty {
  display: grid;
  place-items: center;

  min-height: 190px;

  text-align: center;

  font-size: 0.9em;
  color: var(--muted);
}

.legend {
  margin-top: 8px;

  font-size: 0.8em;
  color: var(--muted);
}

/* =========================================================
   VOTES / QUALITY
   ========================================================= */

.votes {
  grid-column: span 3;
  display: flex;
  flex-direction: column;
}

.voteLine {
  margin-top: auto;
  padding-top: 16px;
  border-top: 1px solid rgba(20, 27, 61, 0.1);

  font-size: 0.9em;
  font-weight: 600;
  line-height: 1.4;
}

.quality {
  grid-column: span 4;
}

.chips {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 8px;
  margin-top: 16px;
  margin-bottom: 24px;
}

.chip {
  display: flex;
  flex-direction: column;
  align-items: center;

  gap: 4px;

  padding: 12px 6px;

  border-radius: 18px;

  background: rgba(255, 255, 255, 0.5);
  border: 1px solid rgba(255, 255, 255, 0.7);
}

.chipN {
  font-size: 0.72em;
  color: var(--muted);
}

.chipV {
  font-size: 1.15em;
  font-weight: 600;
}

.voteLine {
  margin-top: 14px;

  font-size: 0.9em;
  font-weight: 600;
}

.rows {
  margin-top: 14px;
}

.rows > div {
  display: flex;
  justify-content: space-between;
  align-items: center;

  gap: 12px;

  padding: 11px 0;

  border-bottom:
    1px solid rgba(20, 27, 61, 0.1);

  font-size: 0.9em;
}

.rows > div:last-child {
  border-bottom: 0;
}

.rows dt {
  color: var(--muted);
}

.rows dd {
  font-weight: 600;
  text-align: right;
}

.tag {
  display: inline-block;

  padding: 4px 12px;

  border-radius: 999px;

  font-size: 0.85em;
  font-weight: 600;

  background: rgba(91, 123, 255, 0.16);
  color: #2E48B8;
}

/* =========================================================
   RISK BADGES
   ========================================================= */

.risk {
  display: inline-flex;
  align-items: center;

  gap: 6px;

  padding: 5px 12px 5px 9px;

  border-radius: 999px;

  font-size: 0.8em;
  font-weight: 600;

  white-space: nowrap;
}

.risk.low {
  background: #DDF3E9;
  color: #1B6149;
}

.risk.med {
  background: #FFF0D2;
  color: #7D5200;
}

.risk.high {
  background: #FDE0E4;
  color: #9B2540;
}

/* =========================================================
   HISTORY
   ========================================================= */

.history {
  grid-column: span 8;
  grid-row: span 2;

  display: flex;
  flex-direction: column;
}

.filters {
  display: flex;

  gap: 8px;
  margin-top: 16px;
}

.pill {
  height: 32px;

  padding: 0 16px;

  border-radius: 999px;

  font-size: 0.82em;
  font-weight: 600;

  background: rgba(255, 255, 255, 0.4);
  border: 1px solid rgba(255, 255, 255, 0.7);
}

.pill.on {
  background: #0A0A0F;
  border-color: #0A0A0F;

  color: #fff !important;
}

.list {
  list-style: none;

  margin-top: 14px;

  max-height: 340px;

  overflow-y: auto;

  padding-right: 4px;
}

.row {
  width: 100%;

  display: grid;

  grid-template-columns:
    1.4fr
    1.3fr
    .6fr
    .8fr
    16px;

  align-items: center;

  gap: 12px;

  padding: 12px 14px;
  margin-bottom: 8px;

  text-align: left;

  border-radius: 18px;

  border: 1px solid rgba(255, 255, 255, 0.6);

  background: rgba(255, 255, 255, 0.38);

  transition: background 0.2s;
}

.row:hover {
  background: rgba(255, 255, 255, 0.7);
}

.rowDate {
  font-size: 0.9em;
  font-weight: 500;
}

.rowConf {
  font-size: 1.05em;
  font-weight: 600;

  font-variant-numeric: tabular-nums;
}

.rowVote {
  font-size: 0.85em;
  color: var(--muted);
}

.none {
  padding: 24px 0;

  text-align: center;

  font-size: 0.9em;
  color: var(--muted);
}

/* =========================================================
   SEGMENTED CONTROL
   ========================================================= */

.seg {
  display: inline-flex;

  padding: 3px;

  border-radius: 999px;

  background: rgba(255, 255, 255, 0.4);
  border: 1px solid rgba(255, 255, 255, 0.7);
}

.seg button {
  height: 28px;

  padding: 0 13px;

  border: 0;
  border-radius: 999px;

  background: transparent;

  font-size: 0.78em;
  font-weight: 600;

  color: var(--muted);
}

.seg button.on {
  background: #fff;
  color: var(--navy);

  box-shadow:
    0 3px 10px rgba(90, 100, 180, 0.2);
}

/* =========================================================
   ACTIVITY
   ========================================================= */

.activity {
  grid-column: span 4;
}

.actNums {
  display: flex;

  gap: 32px;

  margin-top: 14px;
}

.actNums div {
  display: flex;
  flex-direction: column;
}

.actNums strong {
  font-size: 2.6em;

  font-weight: 300;

  letter-spacing: -0.04em;
  line-height: 1;
}

.actNums span {
  margin-top: 4px;

  font-size: 0.8em;
  color: var(--muted);
}

.bars {
  display: flex;
  align-items: flex-end;
  justify-content: space-around;

  gap: 10px;

  margin-top: 20px;

  height: 150px;
}

.barCol {
  display: flex;
  flex-direction: column;
  align-items: center;

  gap: 6px;
}

.bar {
  width: 44px;

  border-radius: 16px;

  background:
    linear-gradient(
      180deg,
      rgba(91, 123, 255, 0.85),
      rgba(255, 255, 255, 0.65)
    );

  border: 1px solid rgba(255, 255, 255, 0.8);

  box-shadow:
    inset 0 2px 4px rgba(255, 255, 255, 0.6),
    0 8px 18px rgba(91, 123, 255, 0.2);
}

.barN {
  font-size: 0.9em;
  font-weight: 600;
}

.barL {
  font-size: 0.76em;
  color: var(--muted);
}

/* =========================================================
   DATA
   ========================================================= */

.data {
  grid-column: span 4;
}

.dataBtns {
  display: flex;

  flex-wrap: wrap;

  gap: 10px;

  margin-top: 16px;
}

.danger {
  color: #9B2540 !important;
}

/* =========================================================
   WELCOME
   ========================================================= */

.welcome {
  max-width: 560px;
}

.welcome h2,
.mTitle {
  font-size: 1.6em;

  font-weight: 400;

  letter-spacing: -0.03em;
}

.welcome .muted {
  margin-top: 8px;
}

.steps {
  margin: 18px 0 22px 20px !important;

  line-height: 1.8;

  font-size: 0.95em;

  color: var(--muted);
}

/* =========================================================
   MODALS
   ========================================================= */

.scrim {
  position: fixed;
  inset: 0;
  z-index: 50;

  display: grid;
  place-items: center;

  padding: 20px;

  background: rgba(20, 27, 61, 0.32);

  backdrop-filter: blur(6px);
  -webkit-backdrop-filter: blur(6px);
}

.modal {
  position: relative;

  width: min(480px, 100%);

  max-height: 90vh;

  overflow-y: auto;

  padding: 32px;

  border-radius: 28px;

  color: #141B3D;

  font-family:
    Manrope,
    Inter,
    ui-sans-serif,
    sans-serif;

  font-size: 15px;

  background: rgba(244, 246, 255, 0.88);

  border: 1px solid rgba(255, 255, 255, 0.9);

  backdrop-filter: blur(24px);
  -webkit-backdrop-filter: blur(24px);

  box-shadow:
    0 20px 60px rgba(60, 70, 150, 0.35);
}

.modal :where(h2, p, dl, dd, ol) {
  margin: 0;
  padding: 0;
}

.modal .muted,
.modal .legend,
.modal .rows dt,
.modal .steps {
  color: #4F5882;
}

.modal .iconBtn {
  color: #4F5882;
}

.modalClose {
  position: absolute;

  top: 16px;
  right: 16px;
}

.mHead {
  display: flex;
  align-items: center;

  gap: 12px;

  flex-wrap: wrap;

  padding-right: 44px;
}

.mNum {
  margin-top: 16px;

  font-size: 4em;

  font-weight: 300;

  letter-spacing: -0.05em;

  line-height: 1;
}

.mBtns {
  display: flex;

  justify-content: flex-end;

  gap: 10px;

  flex-wrap: wrap;

  margin-top: 22px;
}

.stickyStart {
  display: none;
}

/* =========================================================
   RESPONSIVE
   ========================================================= */

@media (max-width: 1100px) {

  .hero,
  .trend,
  .votes,
  .quality,
  .history,
  .activity,
  .data {
    grid-column: span 12;
    grid-row: auto;
  }

  .votes,
  .quality {
    grid-column: span 6;
  }

  .activity,
  .data {
    grid-column: span 6;
  }
}

@media (max-width: 700px) {

  .page {
    padding: 10px 10px 90px;
  }

  .frame {
    flex-direction: column;

    padding: 16px;

    border-radius: 28px;

    gap: 16px;
  }

  .rail {
    width: 100%;

    flex-direction: row;

    padding: 8px 12px;

    border-radius: 26px;
  }

  .railNav,
  .railEnd {
    flex-direction: row;
    margin: 0;
  }

  .railNav {
    margin-left: 14px;
  }

  .railEnd {
    margin-left: auto;
  }

  .logo {
    width: 38px;
    height: 38px;
  }

  .head h1 {
    font-size: 1.7em;
  }

  .head > .btnDark {
    display: none;
  }

  .votes,
  .quality,
  .activity,
  .data {
    grid-column: span 12;
  }

  .card {
    padding: 22px;
  }

  .heroNum {
    font-size: 4.6em;
  }

  .row {
    grid-template-columns:
      1fr
      auto
      16px;
  }

  .rowConf,
  .rowVote {
    display: none;
  }

  .stickyStart {
    display: flex;

    position: fixed;

    left: 0;
    right: 0;
    bottom: 0;

    z-index: 40;

    padding:
      14px
      16px
      calc(14px + env(safe-area-inset-bottom, 0px));

    background:
      linear-gradient(
        to top,
        rgba(230, 234, 243, 0.95),
        rgba(230, 234, 243, 0)
      );
  }

  .stickyStart .btnDark {
    width: 100%;
  }
}

/* =========================================================
   REDUCED MOTION
   ========================================================= */

@media (prefers-reduced-motion: reduce) {

  .page *,
  .page *::before,
  .page *::after {
    transition: none !important;
  }
}
`;