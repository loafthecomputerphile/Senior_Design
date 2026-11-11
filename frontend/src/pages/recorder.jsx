import React, { useEffect, useState } from "react";

/* ------------------------------------------------------------------ *
 *  Cough capture logic (unchanged behavior). It now lives in this file
 *  and returns a cleanup that also removes its click listeners, so
 *  React StrictMode and route changes can't double-bind or leave the
 *  microphone open.
 * ------------------------------------------------------------------ */

function initCoughCapture(onDone) {
  const CFG = {
    ONSET_DB: -25,
    OFFSET_DB: -35,
    MIN_EVENT_MS: 80,
    MAX_EVENT_MS: 600,
    REFRACTORY_MS: 300,
    BELOW_SUSTAIN_MS: 100,

    METER_MIN_DB: -60,
    METER_MAX_DB: 0,

    SLOT_NUDGE_MS: 6000,
    SESSION_CEILING_MS: 30000,
    CHECK_MS: 1700,
  };

  const els = {
    dots: Array.from(document.querySelectorAll(".rec .dot")),
    fill: document.getElementById("meter-fill"),
    notch: document.getElementById("meter-notch"),
    readout: document.getElementById("readout"),
    readoutValue: document.getElementById("readout-value"),
    heading: document.getElementById("phase-heading"),
    body: document.getElementById("phase-body"),
    primary: document.getElementById("primary"),
    secondary: document.getElementById("secondary"),
  };

  els.notch.style.left =
    (((CFG.ONSET_DB - CFG.METER_MIN_DB) / (CFG.METER_MAX_DB - CFG.METER_MIN_DB)) * 100).toFixed(4) + "%";

  const app = {
    phase: "idle", // idle | requesting | listening | analyzing | done | error
    coughs: 0,
    nudge: false,
    error: null,
    db: CFG.METER_MIN_DB,
    displayDb: CFG.METER_MIN_DB,
  };

  let audioCtx = null;
  let mediaStream = null;
  let graph = null;
  let disposed = false;

  let eventState = { inEvent: false, startedAt: 0, belowSince: null, refractoryUntil: 0 };
  let slotStartedAt = 0;
  let sessionStartedAt = 0;
  let checkTimer = null;
  let finishTimer = null;

  let meterRunning = false;
  let rafId = null;
  let lastPaint = 0;

  /* ---- math ---- */

  function rmsToDb(chunk) {
    let sum = 0;
    for (let i = 0; i < chunk.length; i++) sum += chunk[i] * chunk[i];
    const rms = Math.sqrt(sum / chunk.length);
    return 20 * Math.log10(rms || 1e-10);
  }

  function formatDb(db) {
    const v = Math.round(Math.min(0, Math.max(CFG.METER_MIN_DB, db)));
    return String(v).replace("-", "\u2212");
  }

  /* ---- detection ---- */

  function onChunk(chunk) {
    if (app.phase !== "listening") return;

    const db = rmsToDb(chunk);
    app.db = db;

    const now = performance.now();
    const es = eventState;

    if (!es.inEvent) {
      if (db > CFG.ONSET_DB && now > es.refractoryUntil) {
        es.inEvent = true;
        es.startedAt = now;
        es.belowSince = null;
      }
      return;
    }

    if (db < CFG.OFFSET_DB) {
      if (es.belowSince === null) es.belowSince = now;

      if (now - es.belowSince > CFG.BELOW_SUSTAIN_MS) {
        const duration = es.belowSince - es.startedAt;

        if (duration > CFG.MIN_EVENT_MS && duration < CFG.MAX_EVENT_MS) {
          registerCough();
        }

        es.inEvent = false;
        es.belowSince = null;
        es.refractoryUntil = now + CFG.REFRACTORY_MS;
      }
    } else {
      es.belowSince = null;
    }
  }

  function registerCough() {
    if (app.phase !== "listening" || app.coughs >= 3) return;

    app.coughs += 1;
    app.nudge = false;
    slotStartedAt = performance.now();
    render();

    if (app.coughs >= 3) finishTimer = setTimeout(finish, 90);
  }

  /* ---- audio graph ---- */

  const WORKLET_SRC = `
    class ChunkTap extends AudioWorkletProcessor {
      constructor() {
        super();
        this.buf = new Float32Array(512);
        this.n = 0;
      }
      process(inputs) {
        const ch = inputs[0] && inputs[0][0];
        if (!ch) return true;
        for (let i = 0; i < ch.length; i++) {
          this.buf[this.n++] = ch[i];
          if (this.n === this.buf.length) {
            this.port.postMessage(this.buf.slice(0));
            this.n = 0;
          }
        }
        return true;
      }
    }
    registerProcessor("chunk-tap", ChunkTap);
  `;

  async function buildGraph(ctx, stream) {
    const source = ctx.createMediaStreamSource(stream);
    let node;

    try {
      const url = URL.createObjectURL(new Blob([WORKLET_SRC], { type: "application/javascript" }));
      await ctx.audioWorklet.addModule(url);
      URL.revokeObjectURL(url);

      node = new AudioWorkletNode(ctx, "chunk-tap");
      node.port.onmessage = (e) => onChunk(e.data);
      source.connect(node);
    } catch (err) {
      node = ctx.createScriptProcessor(1024, 1, 1);
      node.onaudioprocess = (e) => onChunk(e.inputBuffer.getChannelData(0));
      source.connect(node);
    }

    const mute = ctx.createGain();
    mute.gain.value = 0;
    node.connect(mute);
    mute.connect(ctx.destination);

    return {
      stop() {
        if (node.port) node.port.onmessage = null;
        if (node.onaudioprocess) node.onaudioprocess = null;
        try { node.disconnect(); } catch (e) {}
        try { mute.disconnect(); } catch (e) {}
        try { source.disconnect(); } catch (e) {}
      },
    };
  }

  /* ---- session lifecycle ---- */

  async function startSession() {
    app.phase = "requesting";
    app.coughs = 0;
    app.nudge = false;
    app.error = null;
    render();

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      fail("unsupported");
      return;
    }

    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      });
    } catch (err) {
      const name = err && err.name;
      if (name === "NotAllowedError" || name === "SecurityError") fail("denied");
      else if (name === "NotFoundError" || name === "OverconstrainedError") fail("nodevice");
      else fail("unknown");
      return;
    }

    // The person may have left the page while the permission prompt was open.
    if (disposed) {
      mediaStream.getTracks().forEach((t) => t.stop());
      mediaStream = null;
      return;
    }

    try {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      await audioCtx.resume();
      graph = await buildGraph(audioCtx, mediaStream);
    } catch (err) {
      fail("unknown");
      return;
    }

    if (disposed) {
      stopAudio();
      return;
    }

    const track = mediaStream.getAudioTracks()[0];
    if (track) {
      track.onended = () => { if (app.phase === "listening") cancel(); };
    }

    app.phase = "listening";
    app.db = CFG.METER_MIN_DB;
    app.displayDb = CFG.METER_MIN_DB;
    eventState = { inEvent: false, startedAt: 0, belowSince: null, refractoryUntil: 0 };

    const now = performance.now();
    slotStartedAt = now;
    sessionStartedAt = now;

    startMeter();
    render();
  }

  function finish() {
    if (app.phase !== "listening") return;
    stopAudio();
    app.phase = "analyzing";
    render();

    checkTimer = setTimeout(() => {
      checkTimer = null;
      app.phase = "done";
      render();
      stopMeter();
      if (onDone) onDone();
    }, CFG.CHECK_MS);
  }

  function cancel() {
    stopAudio();
    stopMeter();
    app.phase = "idle";
    app.coughs = 0;
    app.nudge = false;
    app.db = CFG.METER_MIN_DB;
    app.displayDb = CFG.METER_MIN_DB;
    render();
  }

  function fail(kind) {
    stopAudio();
    stopMeter();
    app.error = kind;
    app.phase = "error";
    app.coughs = 0;
    app.nudge = false;
    app.db = CFG.METER_MIN_DB;
    app.displayDb = CFG.METER_MIN_DB;
    render();
  }

  function stopAudio() {
    if (graph) { graph.stop(); graph = null; }
    if (mediaStream) {
      mediaStream.getTracks().forEach((t) => t.stop());
      mediaStream = null;
    }
    if (audioCtx) { audioCtx.close().catch(() => {}); audioCtx = null; }
    if (checkTimer) { clearTimeout(checkTimer); checkTimer = null; }
    app.db = CFG.METER_MIN_DB;
  }

  /* ---- meter loop ---- */

  function startMeter() {
    if (meterRunning) return;
    meterRunning = true;
    lastPaint = 0;
    rafId = requestAnimationFrame(meterLoop);
  }

  function stopMeter() {
    meterRunning = false;
    if (rafId) cancelAnimationFrame(rafId);
    rafId = null;
  }

  function meterLoop(ts) {
    if (!meterRunning) return;
    rafId = requestAnimationFrame(meterLoop);
    if (ts - lastPaint < 55) return;
    lastPaint = ts;

    const live = app.phase === "listening";
    const target = live ? app.db : CFG.METER_MIN_DB;
    app.displayDb += (target - app.displayDb) * 0.6;
    if (Math.abs(target - app.displayDb) < 0.3) app.displayDb = target;

    paintMeter();

    if (!live) return;

    const now = performance.now();

    if (!app.nudge && now - slotStartedAt > CFG.SLOT_NUDGE_MS) {
      app.nudge = true;
      render();
    }

    if (now - sessionStartedAt > CFG.SESSION_CEILING_MS) {
      fail("timeout");
    }
  }

  function paintMeter() {
    const span = CFG.METER_MAX_DB - CFG.METER_MIN_DB;
    const unit = Math.min(1, Math.max(0, (app.displayDb - CFG.METER_MIN_DB) / span));

    els.fill.style.width = (unit * 100).toFixed(2) + "%";

    const hot = app.displayDb > CFG.ONSET_DB;
    els.fill.classList.toggle("is-hot", hot);
    els.readout.classList.toggle("is-hot", hot);

    const showNumber = app.phase === "listening" || app.phase === "analyzing";
    els.readoutValue.textContent = showNumber ? formatDb(app.displayDb) : "\u2014";
  }

  /* ---- copy ---- */

  const ERRORS = {
    denied: {
      heading: "The microphone is blocked.",
      body: "Allow microphone access for this page in your browser's site settings, then try again.",
    },
    nodevice: {
      heading: "No microphone found.",
      body: "Connect a microphone or a headset, then try again.",
    },
    unsupported: {
      heading: "This browser can't record audio.",
      body: "Open this page in a current version of Chrome, Safari, Edge, or Firefox.",
    },
    timeout: {
      heading: "Nothing came through.",
      body: "The microphone stayed quiet for thirty seconds. Check which microphone is selected, then try again.",
    },
    unknown: {
      heading: "The microphone failed to start.",
      body: "Close other apps that might be using it, then try again.",
    },
  };

  function copyFor(phase, coughs, nudge) {
    switch (phase) {
      case "idle":
        return {
          heading: "Give three coughs.",
          body: "The microphone opens first so you can check it's hearing you. Then cough three times, one at a time, as loud and clear as you can.",
        };
      case "requesting":
        return {
          heading: "Waiting for microphone access.",
          body: "Your browser is asking whether this page can use the microphone. Choose Allow.",
        };
      case "listening": {
        const slot = coughs + 1;
        let body;
        if (nudge) body = "Nothing's registered yet. Move closer to the microphone, or cough a little harder.";
        else if (slot === 1) body = "Say a few words to check the bar moves. Then give a loud, clear cough.";
        else if (slot === 2) body = "Take a breath, then cough again.";
        else body = "Last one. Loud and clear.";
        return { heading: "Cough " + slot + " of 3", body };
      }
      case "analyzing":
        return { heading: "All three captured.", body: "Checking that all three coughs came through clearly." };
      case "done":
        return { heading: "Sample captured.", body: "Three coughs recorded. You can record another sample at any time." };
      case "error": {
        const e = ERRORS[app.error] || ERRORS.unknown;
        return { heading: e.heading, body: e.body };
      }
      default:
        return { heading: "", body: "" };
    }
  }

  function buttonsFor(phase) {
    switch (phase) {
      case "idle":       return { primary: { label: "Turn on microphone" }, secondary: null };
      case "requesting": return { primary: { label: "Waiting for permission", disabled: true }, secondary: null };
      case "listening":  return { primary: null, secondary: { label: "Stop" } };
      case "analyzing":  return { primary: null, secondary: null };
      case "done":       return { primary: { label: "Record another" }, secondary: null };
      case "error":      return { primary: { label: "Try again" }, secondary: null };
      default:           return { primary: null, secondary: null };
    }
  }

  /* ---- render ---- */

  function render() {
    const { phase, coughs, nudge } = app;

    els.dots.forEach((dot, i) => {
      let state = "pending";
      if (i < coughs) state = "captured";
      else if (i === coughs && phase === "listening") state = "active";
      if (dot.dataset.state !== state) dot.dataset.state = state;
    });

    const copy = copyFor(phase, coughs, nudge);
    if (els.heading.textContent !== copy.heading) els.heading.textContent = copy.heading;
    if (els.body.textContent !== copy.body) els.body.textContent = copy.body;

    const btns = buttonsFor(phase);

    if (btns.primary) {
      els.primary.hidden = false;
      els.primary.textContent = btns.primary.label;
      els.primary.disabled = !!btns.primary.disabled;
    } else {
      els.primary.hidden = true;
    }

    if (btns.secondary) {
      els.secondary.hidden = false;
      els.secondary.textContent = btns.secondary.label;
    } else {
      els.secondary.hidden = true;
    }

    paintMeter();
  }

  /* ---- wiring ---- */

  const onPrimary = () => {
    if (app.phase === "idle" || app.phase === "done" || app.phase === "error") startSession();
  };
  const onSecondary = () => {
    if (app.phase === "listening") cancel();
  };

  els.primary.addEventListener("click", onPrimary);
  els.secondary.addEventListener("click", onSecondary);

  render();

  return () => {
    disposed = true;
    els.primary.removeEventListener("click", onPrimary);
    els.secondary.removeEventListener("click", onSecondary);
    if (finishTimer) clearTimeout(finishTimer);
    try { cancel(); } catch (e) {}
  };
}

/* ---- icons ---- */

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
  lung: () => (
    <Svg>
      <path d="M12 4v7M12 11c-1.5 1-2.5 1.5-3.5 1.5M12 11c1.5 1 2.5 1.5 3.5 1.5" />
      <path d="M8.5 8C6 9 4 14 4 17.5 4 19 5 20 6.3 20c1.6 0 2.7-1 2.7-2.6V8.3ZM15.5 8C18 9 20 14 20 17.5c0 1.5-1 2.5-2.3 2.5-1.6 0-2.7-1-2.7-2.6V8.3Z" />
    </Svg>
  ),
  info: () => (
    <Svg size={16}>
      <circle cx="12" cy="12" r="8.5" />
      <path d="M12 11v5M12 7.9v.1" />
    </Svg>
  ),
  back: () => (
    <Svg size={16}>
      <path d="m14.5 6-6 6 6 6" />
    </Svg>
  ),
  close: () => (
    <Svg size={18}>
      <path d="m6.5 6.5 11 11M17.5 6.5l-11 11" />
    </Svg>
  ),
};

const DISCLAIMER = "Pre-screening tool only, not a diagnosis.";

/* ---- page ---- */

export default function Recorder({ onBack }) {
  const [showHelp, setShowHelp] = useState(false);
  const [done, setDone] = useState(false);

  useEffect(() => {
    const cleanup = initCoughCapture(() => setDone(true));
    return cleanup;
  }, []);

  // Escape closes the help dialog.
  useEffect(() => {
    if (!showHelp) return;
    const onKey = (e) => e.key === "Escape" && setShowHelp(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [showHelp]);

  return (
    <>
      <style>{CSS}</style>

      <div className="rec page">
        <div className="frame">
          {/* ---------- slim icon sidebar (mirrors the dashboard) ---------- */}
          <aside className="rail" aria-label="Navigation">
            <div className="logo" aria-hidden="true"><I.lung /></div>
            <nav className="railNav">
              <button type="button" className="iconBtn" aria-label="Dashboard" title="Dashboard" onClick={onBack}><I.grid /></button>
              <button type="button" className="iconBtn active" aria-label="Recorder" aria-current="page" title="Recorder"><I.mic /></button>
              <button type="button" className="iconBtn" aria-label="How it works" title="How it works" onClick={() => setShowHelp(true)}><I.help /></button>
            </nav>
          </aside>

          {/* ---------- main ---------- */}
          <main className="main">
            <header className="head">
              <div>
                <h1>Cough test</h1>
                <p className="sub">Three coughs, under a minute</p>
              </div>
              <button type="button" className="btnGlass" onClick={onBack}>
                <I.back /> Back to dashboard
              </button>
            </header>

            <div className="strip" role="note">
              <I.info /> {DISCLAIMER} Your audio is not saved.
            </div>

            <div className="stage">
              {/* Instrument: vivid hero card */}
              <section className="card hero" aria-hidden="true">
                <div className="instrumentHead">
                  <ol className="dots">
                    {[0, 1, 2].map((i) => (
                      <li key={i} className="dot" data-state="pending">
                        <span className="dot-halo"></span>
                        <span className="dot-fill"></span>
                      </li>
                    ))}
                  </ol>
                  <div className="readout" id="readout">
                    <span className="readout-value" id="readout-value">—</span>
                    <span className="readout-unit">dB</span>
                  </div>
                </div>

                <div className="meter">
                  <div className="meter-track">
                    <div className="meter-fill" id="meter-fill"></div>
                    <div className="meter-notch" id="meter-notch"></div>
                  </div>
                  <div className="meter-scale">
                    <span>−60</span><span>−40</span><span>−20</span><span>0</span>
                  </div>
                </div>
              </section>

              {/* Script + controls: glass card */}
              <section className="card script">
                <div aria-live="polite" aria-atomic="true">
                  <h2 id="phase-heading">Give three coughs.</h2>
                  <p id="phase-body">The microphone opens first so you can check it's hearing you. Then cough three times, one at a time, as loud and clear as you can.</p>
                </div>

                <div className="controls">
                  <button type="button" id="primary" className="btnDark">Turn on microphone</button>
                  <button type="button" id="secondary" className="btnGlass" hidden>Stop</button>
                  {done && (
                    <button type="button" className="btnGlass" onClick={onBack}>View dashboard</button>
                  )}
                </div>
              </section>
            </div>
          </main>
        </div>
      </div>

      {showHelp && (
        <div className="rec scrim" onMouseDown={() => setShowHelp(false)}>
          <div className="modal" role="dialog" aria-modal="true" aria-label="How it works" onMouseDown={(e) => e.stopPropagation()}>
            <button type="button" className="iconBtn modalClose" aria-label="Close" onClick={() => setShowHelp(false)}><I.close /></button>
            <h2 className="mTitle">How it works</h2>
            <ol className="steps">
              <li>Press Turn on microphone and allow access.</li>
              <li>Check the bar moves when you speak.</li>
              <li>Give three loud, clear coughs, one at a time.</li>
              <li>We check the audio, then you can return to your dashboard.</li>
            </ol>
            <p className="legend">{DISCLAIMER} Confirm any concern with an official test.</p>
          </div>
        </div>
      )}
    </>
  );
}

/* ---- styles: same tokens as the dashboard ---- */

const CSS = `
@import url("https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;500;600;700&display=swap");

.rec, .rec * { box-sizing: border-box; }

.rec.page {
  --navy: #141B3D;
  --muted: #4F5882;
  --accent: #5B7BFF;
  --glass: rgba(255, 255, 255, 0.38);
  --glass-line: rgba(255, 255, 255, 0.55);
  --shadow: 0 10px 40px rgba(90, 100, 180, 0.15);

  min-height: 100vh;
  padding: 24px;
  background: #E6EAF3;
  font-family: Manrope, Inter, ui-sans-serif, -apple-system, "Segoe UI", sans-serif;
  font-size: 15px;
  color: var(--navy);
  -webkit-font-smoothing: antialiased;
}

.rec [hidden] { display: none !important; }
.rec button { font: inherit; color: inherit; cursor: pointer; }
.rec button:disabled { cursor: not-allowed; opacity: .45; }
.rec button:focus-visible { outline: 2px solid var(--navy); outline-offset: 2px; }
.rec h1, .rec h2, .rec p, .rec ol, .rec ul { margin: 0; padding: 0; }

/* frame */
.rec .frame {
  display: flex; gap: 28px;
  min-height: calc(100vh - 48px);
  padding: 28px;
  border-radius: 40px;
  background:
    radial-gradient(circle at 15% 20%, rgba(120,145,255,.75), transparent 40%),
    radial-gradient(circle at 85% 30%, rgba(170,155,255,.65), transparent 45%),
    linear-gradient(135deg, #AEBBE5 0%, #C4CCE9 45%, #D9DDEB 100%);
  box-shadow: 0 30px 80px rgba(90,100,180,.22);
}

/* sidebar */
.rec .rail {
  width: 64px; flex-shrink: 0;
  display: flex; flex-direction: column; align-items: center;
  padding: 18px 0; border-radius: 32px;
  background: var(--glass); border: 1px solid var(--glass-line);
  backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
  box-shadow: var(--shadow);
}
.rec .logo {
  width: 44px; height: 44px; border-radius: 50%;
  display: grid; place-items: center; color: #fff;
  background: linear-gradient(135deg, #5B7BFF, #7B6CF6);
  box-shadow: 0 8px 20px rgba(91,123,255,.35);
}
.rec .railNav { display: flex; flex-direction: column; gap: 10px; margin-top: 36px; }
.rec .iconBtn {
  width: 44px; height: 44px; border-radius: 50%;
  display: grid; place-items: center;
  border: 1px solid transparent; background: transparent; color: var(--muted);
  transition: background .2s, color .2s;
}
.rec .iconBtn:hover:not(:disabled) { background: rgba(255,255,255,.5); color: var(--navy); }
.rec .iconBtn.active { background: #fff; color: var(--navy); box-shadow: 0 6px 16px rgba(90,100,180,.2); }

/* main */
.rec .main { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 18px; }
.rec .head {
  display: flex; align-items: center; justify-content: space-between;
  gap: 16px; flex-wrap: wrap; padding: 4px 4px 0;
}
.rec .head h1 { font-size: 2.1em; font-weight: 300; letter-spacing: -.035em; line-height: 1.1; }
.rec .sub { margin-top: 6px; font-size: .9em; color: var(--muted); }

.rec .strip {
  display: flex; align-items: center; gap: 8px; align-self: flex-start;
  padding: 9px 16px; border-radius: 999px;
  font-size: .82em; color: var(--muted);
  background: var(--glass); border: 1px solid var(--glass-line);
  backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
}
.rec .strip svg { flex-shrink: 0; }

/* buttons */
.rec .btnDark {
  display: inline-flex; align-items: center; justify-content: center; gap: 8px;
  height: 46px; padding: 0 24px; border: 0; border-radius: 999px;
  background: #0A0A0F; color: #fff !important;
  font-size: .95em; font-weight: 600;
  box-shadow: 0 10px 24px rgba(10,10,15,.25);
  transition: transform .15s;
}
.rec .btnDark:hover:not(:disabled) { transform: translateY(-1px); }
.rec .btnGlass {
  display: inline-flex; align-items: center; justify-content: center; gap: 8px;
  height: 46px; padding: 0 20px; border-radius: 999px;
  border: 1px solid rgba(255,255,255,.7); background: rgba(255,255,255,.55);
  font-size: .9em; font-weight: 600; transition: background .2s;
}
.rec .head .btnGlass { height: 40px; font-size: .86em; }
.rec .btnGlass:hover:not(:disabled) { background: rgba(255,255,255,.85); }

/* cards */
.rec .stage {
  flex: 1;
  display: grid; grid-template-columns: repeat(12, minmax(0, 1fr)); gap: 20px;
  align-content: start;
}
.rec .card {
  position: relative; padding: 26px; border-radius: 24px; min-width: 0;
  background: var(--glass); border: 1px solid var(--glass-line);
  backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
  box-shadow: var(--shadow);
}
.rec .hero {
  grid-column: span 7;
  display: flex; flex-direction: column; justify-content: space-between; gap: 40px;
  min-height: 340px;
  background:
    radial-gradient(circle at 20% 10%, rgba(255,255,255,.25), transparent 35%),
    linear-gradient(145deg, #627FFF 0%, #796CF4 35%, #A5B5FF 65%, #DCE4FF 100%);
  color: #fff;
}
.rec .script {
  grid-column: span 5;
  display: flex; flex-direction: column; justify-content: space-between; gap: 28px;
}
.rec .script h2 { font-size: 1.9em; font-weight: 300; letter-spacing: -.035em; line-height: 1.15; }
.rec .script p { margin-top: 12px; line-height: 1.6; color: var(--muted); max-width: 46ch; }
.rec .controls { display: flex; flex-wrap: wrap; gap: 10px; }

/* instrument head */
.rec .instrumentHead { display: flex; align-items: center; justify-content: space-between; gap: 16px; }
.rec .dots { list-style: none; display: flex; gap: 14px; }
.rec .dot { position: relative; width: 48px; height: 48px; }
.rec .dot-halo, .rec .dot-fill { position: absolute; border-radius: 50%; }
.rec .dot-halo {
  inset: 0;
  background: rgba(255,255,255,.18);
  border: 1px solid rgba(255,255,255,.5);
  transition: background .3s, border-color .3s;
}
.rec .dot-fill {
  inset: 10px; background: #fff; transform: scale(0);
  box-shadow: 0 6px 16px rgba(20,27,61,.25);
  transition: transform .35s cubic-bezier(.3,1.4,.5,1);
}
.rec .dot[data-state="active"] .dot-halo {
  background: rgba(255,255,255,.35); border-color: #fff;
  animation: recPulse 1.4s ease-in-out infinite;
}
.rec .dot[data-state="captured"] .dot-halo { background: rgba(255,255,255,.4); border-color: #fff; }
.rec .dot[data-state="captured"] .dot-fill { transform: scale(1); }
@keyframes recPulse {
  0%, 100% { box-shadow: 0 0 0 0 rgba(255,255,255,.55); }
  50%      { box-shadow: 0 0 0 10px rgba(255,255,255,0); }
}

/* readout */
.rec .readout { display: flex; align-items: baseline; gap: 6px; color: rgba(255,255,255,.8); transition: color .15s; }
.rec .readout-value {
  font-size: 3.4em; font-weight: 300; letter-spacing: -.05em; line-height: 1;
  font-variant-numeric: tabular-nums; min-width: 2.2ch; text-align: right;
}
.rec .readout-unit { font-size: .95em; }
.rec .readout.is-hot { color: #fff; }

/* meter */
.rec .meter-track {
  position: relative; height: 18px; border-radius: 999px;
  background: rgba(255,255,255,.28);
  box-shadow: inset 0 2px 4px rgba(20,27,61,.12);
}
.rec .meter-fill {
  height: 100%; width: 0; border-radius: 999px;
  background: rgba(255,255,255,.7);
  box-shadow: inset 0 2px 3px rgba(255,255,255,.6);
  transition: width 90ms linear, background .15s, box-shadow .15s;
}
.rec .meter-fill.is-hot {
  background: #fff;
  box-shadow: inset 0 2px 3px rgba(255,255,255,.8), 0 0 18px rgba(255,255,255,.7);
}
.rec .meter-notch {
  position: absolute; top: -5px; bottom: -5px; width: 2px;
  background: var(--navy); opacity: .55; border-radius: 2px;
}
.rec .meter-scale {
  display: flex; justify-content: space-between; margin-top: 10px;
  font-size: .76em; color: rgba(255,255,255,.85); font-variant-numeric: tabular-nums;
}

/* help dialog */
.rec.scrim {
  position: fixed; inset: 0; z-index: 50; display: grid; place-items: center; padding: 20px;
  background: rgba(20,27,61,.32);
  backdrop-filter: blur(6px); -webkit-backdrop-filter: blur(6px);
  min-height: 0;
}
.rec .modal {
  position: relative; width: min(480px, 100%); max-height: 90vh; overflow-y: auto;
  padding: 32px; border-radius: 28px; color: #141B3D;
  font-family: Manrope, Inter, ui-sans-serif, sans-serif; font-size: 15px;
  background: rgba(244,246,255,.88); border: 1px solid rgba(255,255,255,.9);
  backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px);
  box-shadow: 0 20px 60px rgba(60,70,150,.35);
}
.rec .modal .iconBtn { color: #4F5882; }
.rec .modalClose { position: absolute; top: 16px; right: 16px; }
.rec .mTitle { font-size: 1.6em; font-weight: 400; letter-spacing: -.03em; padding-right: 44px; }
.rec .steps { margin: 18px 0 22px 20px !important; line-height: 1.8; font-size: .95em; color: #4F5882; }
.rec .legend { font-size: .8em; color: #4F5882; }

/* responsive */
@media (max-width: 1100px) {
  .rec .hero, .rec .script { grid-column: span 12; }
}
@media (max-width: 700px) {
  .rec.page { padding: 10px; }
  .rec .frame { flex-direction: column; padding: 16px; border-radius: 28px; gap: 16px; }
  .rec .rail { width: 100%; flex-direction: row; padding: 8px 12px; border-radius: 26px; }
  .rec .railNav { flex-direction: row; margin: 0 0 0 14px; }
  .rec .logo { width: 38px; height: 38px; }
  .rec .head h1 { font-size: 1.7em; }
  .rec .card { padding: 22px; }
  .rec .readout-value { font-size: 2.8em; }
  .rec .controls .btnDark, .rec .controls .btnGlass { flex: 1; }
}

@media (prefers-reduced-motion: reduce) {
  .rec *, .rec *::before, .rec *::after { transition: none !important; animation: none !important; }
}
`;
