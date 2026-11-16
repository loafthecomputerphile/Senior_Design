import { useEffect, useState } from "react";
import Dashboard from "./pages/dashboard";
import Recorder from "./pages/recorder";
import SignIn, { loadSession, clearSession } from "./pages/signin";

const routeFromHash = () => {
  const h = window.location.hash;
  if (h === "#/signin") return "signin";
  if (h === "#/recorder") return "recorder";
  return "dashboard";
};

const HASH = { signin: "#/signin", recorder: "#/recorder", dashboard: "#/" };

export default function App() {
  const [session, setSession] = useState(loadSession);
  const [route, setRoute] = useState(routeFromHash);

  // Keep state in sync with the URL (back/forward buttons, manual hash edits).
  useEffect(() => {
    const onHash = () => setRoute(routeFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  // Signed-out visitors only ever see the sign-in page;
  // signed-in visitors skip it.
  const view = !session ? "signin" : route === "signin" ? "dashboard" : route;

  // Keep the address bar honest without adding history entries.
  useEffect(() => {
    const h = window.location.hash;
    const ok = view === "dashboard" ? h === "" || h === "#/" : h === HASH[view];
    if (!ok) window.history.replaceState(null, "", HASH[view]);
  }, [view]);

  const go = (to) => {
    window.location.hash = HASH[to];
    window.scrollTo(0, 0);
  };

  const handleSignedIn = (s) => {
    setSession(s);
    go("dashboard");
  };

  const handleSignOut = () => {
    clearSession();
    setSession(null);
  };

  if (view === "signin") return <SignIn onSignedIn={handleSignedIn} />;
  if (view === "recorder") return <Recorder onBack={() => go("dashboard")} />;

  // Every "Start cough test" / mic button in the dashboard calls onStartTest.
  return (
    <Dashboard
      user={session.user}
      onSignOut={handleSignOut}
      onStartTest={() => go("recorder")}
    />
  );
}