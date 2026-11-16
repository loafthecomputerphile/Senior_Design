import React, { useEffect, useRef, useState } from "react";

/* ------------------------------------------------------------------ *
 *  Config (put these in a .env file at the project root, then restart
 *  the dev server):
 *    VITE_GOOGLE_CLIENT_ID=xxxxxxxx.apps.googleusercontent.com
 *    VITE_API_URL=http://localhost:8000        (optional, see below)
 *
 *  With VITE_API_URL set, the Google credential is sent to
 *  POST {API}/auth/google, which must verify it and return
 *  { access_token, user? }.
 *  Without it, the page runs in client-only mode: it checks the token's
 *  audience and expiry and reads the profile from it. That is fine for
 *  building the UI, but it is NOT verified, so nothing sensitive may
 *  rely on it.
 * ------------------------------------------------------------------ */

const CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID;
const API = import.meta.env.VITE_API_URL;
const DEV = import.meta.env.DEV;
const DISCLAIMER = "Pre-screening tool only, not a diagnosis.";

/* ---- session helpers (used by App.jsx) ----------------------------- */

const KEY = "cough.session";

export function loadSession() {
  try {
    const raw = sessionStorage.getItem(KEY);
    return raw ? JSON.parse(raw) : null;
  } catch (e) {
    return null;
  }
}

function saveSession(s) {
  try { sessionStorage.setItem(KEY, JSON.stringify(s)); } catch (e) {}
}

export function clearSession() {
  try { sessionStorage.removeItem(KEY); } catch (e) {}
  try { window.google?.accounts?.id?.disableAutoSelect?.(); } catch (e) {}
}

/* ---- google script + token helpers --------------------------------- */

let gsiPromise = null;

function loadGoogleScript() {
  if (window.google?.accounts?.id) return Promise.resolve();
  if (!gsiPromise) {
    gsiPromise = new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = "https://accounts.google.com/gsi/client";
      s.async = true;
      s.defer = true;
      s.onload = resolve;
      s.onerror = () => {
        gsiPromise = null;
        reject(new Error("script"));
      };
      document.head.appendChild(s);
    });
  }
  return gsiPromise;
}

function decodeJwt(token) {
  const b64 = token.split(".")[1].replace(/-/g, "+").replace(/_/g, "/");
  const json = decodeURIComponent(
    atob(b64)
      .split("")
      .map((c) => "%" + ("00" + c.charCodeAt(0).toString(16)).slice(-2))
      .join("")
  );
  return JSON.parse(json);
}

function profileFrom(payload) {
  return {
    sub: payload.sub,
    name: payload.name || payload.email,
    email: payload.email,
    picture: payload.picture || null,
  };
}

async function exchangeCredential(credential) {
  const payload = decodeJwt(credential);

  if (API) {
    let res;
    try {
      res = await fetch(`${API}/auth/google`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ credential }),
      });
    } catch (e) {
      throw new Error("Can't reach the server. Check your connection and try again.");
    }
    if (res.status === 401) throw new Error("Google sign-in was rejected. Try again.");
    if (!res.ok) throw new Error("The server couldn't sign you in. Try again in a moment.");
    const data = await res.json();
    return { token: data.access_token, user: data.user || profileFrom(payload) };
  }

  // Client-only mode: basic sanity checks, no signature verification.
  const issuerOk = payload.iss === "https://accounts.google.com" || payload.iss === "accounts.google.com";
  if (payload.aud !== CLIENT_ID || !issuerOk || payload.exp * 1000 < Date.now()) {
    throw new Error("That sign-in has expired or isn't valid. Try again.");
  }
  return { token: null, user: profileFrom(payload) };
}

/* ---- icons --------------------------------------------------------- */

const Svg = ({ children, size = 20 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    {children}
  </svg>
);

const LungIcon = () => (
  <Svg>
    <path d="M12 4v7M12 11c-1.5 1-2.5 1.5-3.5 1.5M12 11c1.5 1 2.5 1.5 3.5 1.5" />
    <path d="M8.5 8C6 9 4 14 4 17.5 4 19 5 20 6.3 20c1.6 0 2.7-1 2.7-2.6V8.3ZM15.5 8C18 9 20 14 20 17.5c0 1.5-1 2.5-2.3 2.5-1.6 0-2.7-1-2.7-2.6V8.3Z" />
  </Svg>
);

const CheckIcon = () => (
  <Svg size={16}>
    <circle cx="12" cy="12" r="8.5" />
    <path d="m8.5 12.3 2.4 2.4 4.6-5" />
  </Svg>
);

/* ---- page ---------------------------------------------------------- */

export default function SignIn({ onSignedIn }) {
  const btnRef = useRef(null);
  const onSignedInRef = useRef(onSignedIn);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    onSignedInRef.current = onSignedIn;
  }, [onSignedIn]);

  useEffect(() => {
    if (!CLIENT_ID) return;
    let cancelled = false;

    async function handleCredential(resp) {
      setError("");
      setBusy(true);
      try {
        const session = await exchangeCredential(resp.credential);
        saveSession(session);
        onSignedInRef.current(session);
      } catch (e) {
        if (!cancelled) {
          setError(e.message || "Sign-in failed. Try again.");
          setBusy(false);
        }
      }
    }

    loadGoogleScript()
      .then(() => {
        if (cancelled || !btnRef.current) return;
        window.google.accounts.id.initialize({
          client_id: CLIENT_ID,
          callback: handleCredential,
          auto_select: false,
        });
        const width = Math.max(200, Math.min(360, btnRef.current.offsetWidth || 320));
        window.google.accounts.id.renderButton(btnRef.current, {
          type: "standard",
          theme: "outline",
          size: "large",
          shape: "pill",
          text: "continue_with",
          logo_alignment: "left",
          width,
        });
      })
      .catch(() => {
        if (!cancelled) setError("Couldn't load Google sign-in. Check your connection and refresh the page.");
      });

    return () => {
      cancelled = true;
      try { window.google?.accounts?.id?.cancel?.(); } catch (e) {}
    };
  }, []);

  const continueDemo = () => {
    const session = { token: null, user: { sub: "demo", name: "Demo user", email: "demo@local", picture: null } };
    saveSession(session);
    onSignedIn(session);
  };

  return (
    <>
      <style>{CSS}</style>

      <div className="auth page">
        <div className="frame">
          <div className="stage">
            {/* Brand / reassurance: vivid hero card */}
            <section className="card hero" aria-labelledby="a-title">
              <div className="logo" aria-hidden="true"><LungIcon /></div>

              <div className="heroText">
                <h1 id="a-title">Track your cough results over time.</h1>
                <p>Record three coughs, get a risk level, and see how it changes from week to week.</p>
              </div>

              <ul className="heroPanel">
                <li><CheckIcon /> Your results stay on this device.</li>
                <li><CheckIcon /> Your recordings are never saved.</li>
                <li><CheckIcon /> {DISCLAIMER}</li>
              </ul>
            </section>

            {/* Sign in: glass card */}
            <section className="card panel" aria-labelledby="a-signin">
              <div>
                <h2 id="a-signin">Sign in</h2>
                <p className="lead">Continue with your Google account to see your history and start a test.</p>
              </div>

              <div className="action">
                {CLIENT_ID ? (
                  <div className="gbtn" ref={btnRef} aria-busy={busy} />
                ) : (
                  <div className="notice" role="note">
                    Google sign-in isn't set up yet. Add <code>VITE_GOOGLE_CLIENT_ID</code> to your <code>.env</code> file, then restart the dev server.
                  </div>
                )}

                {busy && <p className="status" role="status">Signing you in…</p>}
                {error && <p className="error" role="alert">{error}</p>}

                {!CLIENT_ID && DEV && (
                  <button type="button" className="btnGlass" onClick={continueDemo}>Continue in demo mode</button>
                )}
              </div>

              <p className="legal">We only read your name and email from Google, to show who is signed in.</p>
            </section>
          </div>
        </div>
      </div>
    </>
  );
}

/* ---- styles: same tokens as the dashboard and recorder -------------- */

const CSS = `
@import url("https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;500;600;700&display=swap");

.auth, .auth * { box-sizing: border-box; }

.auth.page {
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

.auth [hidden] { display: none !important; }
.auth button { font: inherit; color: inherit; cursor: pointer; }
.auth button:focus-visible { outline: 2px solid var(--navy); outline-offset: 2px; }

/* :where() keeps this reset at zero specificity so class margins still win */
.auth :where(h1, h2, p, ul) { margin: 0; padding: 0; }

.auth .frame {
  display: flex;
  align-items: center;
  min-height: calc(100vh - 48px);
  padding: 28px;
  border-radius: 40px;
  background:
    radial-gradient(circle at 15% 20%, rgba(120,145,255,.75), transparent 40%),
    radial-gradient(circle at 85% 30%, rgba(170,155,255,.65), transparent 45%),
    linear-gradient(135deg, #AEBBE5 0%, #C4CCE9 45%, #D9DDEB 100%);
  box-shadow: 0 30px 80px rgba(90,100,180,.22);
}

.auth .stage {
  width: 100%;
  max-width: 1040px;
  margin: 0 auto;
  display: grid;
  grid-template-columns: repeat(12, minmax(0, 1fr));
  gap: 20px;
}

.auth .card {
  padding: 32px;
  border-radius: 24px;
  min-width: 0;
  background: var(--glass);
  border: 1px solid var(--glass-line);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  box-shadow: var(--shadow);
}

/* hero */
.auth .hero {
  grid-column: span 7;
  display: flex;
  flex-direction: column;
  gap: 40px;
  min-height: 460px;
  color: #fff;
  background:
    radial-gradient(circle at 20% 10%, rgba(255,255,255,.25), transparent 35%),
    linear-gradient(145deg, #627FFF 0%, #796CF4 35%, #A5B5FF 65%, #DCE4FF 100%);
}
.auth .logo {
  width: 44px; height: 44px; border-radius: 50%;
  display: grid; place-items: center;
  color: var(--navy);
  background: rgba(255,255,255,.85);
  box-shadow: 0 8px 20px rgba(20,27,61,.18);
}
.auth .heroText h1 {
  max-width: 14ch;
  font-size: 3em; font-weight: 300; letter-spacing: -.04em; line-height: 1.05;
}
.auth .heroText p {
  max-width: 40ch; margin-top: 16px;
  font-size: 1em; line-height: 1.6; color: rgba(255,255,255,.9);
}
.auth .heroPanel {
  list-style: none;
  margin-top: auto;
  padding: 20px 22px;
  display: flex; flex-direction: column; gap: 12px;
  border-radius: 20px;
  color: var(--navy);
  font-size: .92em;
  background: rgba(255,255,255,.55);
  border: 1px solid rgba(255,255,255,.8);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
}
.auth .heroPanel li { display: flex; align-items: center; gap: 10px; }
.auth .heroPanel svg { flex-shrink: 0; color: var(--accent); }

/* sign in panel */
.auth .panel {
  grid-column: span 5;
  display: flex; flex-direction: column; justify-content: space-between; gap: 32px;
}
.auth .panel h2 { font-size: 2em; font-weight: 300; letter-spacing: -.035em; line-height: 1.1; }
.auth .lead { margin-top: 12px; line-height: 1.6; color: var(--muted); }

.auth .action { display: flex; flex-direction: column; align-items: stretch; gap: 14px; }
.auth .gbtn { min-height: 44px; display: flex; justify-content: center; }
.auth .gbtn[aria-busy="true"] { opacity: .5; pointer-events: none; }

.auth .notice {
  padding: 14px 16px; border-radius: 18px;
  font-size: .88em; line-height: 1.55; color: var(--navy);
  background: rgba(255,255,255,.6);
  border: 1px solid rgba(255,255,255,.8);
}
.auth .notice code {
  padding: 1px 6px; border-radius: 6px;
  font-size: .92em; background: rgba(91,123,255,.14);
}
.auth .status { font-size: .88em; color: var(--muted); text-align: center; }
.auth .error {
  padding: 10px 14px; border-radius: 16px;
  font-size: .88em; line-height: 1.5;
  background: #FDE0E4; color: #9B2540;
}

.auth .btnGlass {
  display: inline-flex; align-items: center; justify-content: center;
  height: 46px; padding: 0 20px; border-radius: 999px;
  border: 1px solid rgba(255,255,255,.7); background: rgba(255,255,255,.55);
  font-size: .9em; font-weight: 600; transition: background .2s;
}
.auth .btnGlass:hover { background: rgba(255,255,255,.85); }

.auth .legal { font-size: .8em; line-height: 1.5; color: var(--muted); }

/* responsive */
@media (max-width: 900px) {
  .auth .hero, .auth .panel { grid-column: span 12; }
  .auth .hero { min-height: 0; }
  .auth .heroText h1 { font-size: 2.3em; }
}
@media (max-width: 700px) {
  .auth.page { padding: 10px; }
  .auth .frame { padding: 16px; border-radius: 28px; }
  .auth .card { padding: 24px; }
}

@media (prefers-reduced-motion: reduce) {
  .auth *, .auth *::before, .auth *::after { transition: none !important; animation: none !important; }
}
`;
