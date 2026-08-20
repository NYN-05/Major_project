import { useEffect, useState } from "react";

export function useTheme() {
  const [theme, setTheme] = useState(() => {
    try {
      const saved = localStorage.getItem("rppgqc.theme");
      if (saved === "dark" || saved === "light") return saved;
    } catch {
      /* storage unavailable */
    }
    return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  });

  useEffect(() => {
    document.documentElement.classList.toggle("light", theme === "light");
    try {
      localStorage.setItem("rppgqc.theme", theme);
    } catch {
      /* storage unavailable */
    }
  }, [theme]);

  return [theme, setTheme];
}

export function useElapsed(active) {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    if (!active) return undefined;
    const t0 = Date.now();
    const id = setInterval(() => setElapsed((Date.now() - t0) / 1000), 1000);
    return () => clearInterval(id);
  }, [active]);
  return elapsed;
}

export function useSignalFile(fileUrl, rel) {
  const [signal, setSignal] = useState(null);
  useEffect(() => {
    let alive = true;
    if (!rel) {
      setSignal(null);
      return undefined;
    }
    fetch(fileUrl(rel))
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
      .then((data) => alive && setSignal(data))
      .catch(() => alive && setSignal({ signal: null, error: "unavailable" }));
    return () => {
      alive = false;
    };
  }, [fileUrl, rel]);
  return signal;
}