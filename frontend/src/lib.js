import { Atom, Clapperboard, Gauge, HeartPulse, ScanFace, ScanSearch, ShieldCheck } from "lucide-react";

export const PLAIN = {
  REAL: {
    word: "Likely real",
    icon: "check",
    tone: "ok",
    note: "This video shows a real, live person. The natural pulse of the face matches a genuine recording.",
  },
  FAKE: {
    word: "Likely deepfake",
    icon: "x",
    tone: "bad",
    note: "This video does not show the natural pulse of a live person — typical of AI-generated or manipulated footage.",
  },
  UNCERTAIN: {
    word: "Needs human review",
    icon: "warn",
    tone: "warn",
    note: "The signal was too weak for a confident decision. A manual review is recommended before this video is used.",
  },
};

export const PIPELINE = [
  { icon: Clapperboard, title: "Video received", sub: "Reading the uploaded clip" },
  { icon: ScanSearch, title: "Frame quality", sub: "Filtering blurry or dark frames" },
  { icon: ScanFace, title: "Face detection", sub: "Locating the face region" },
  { icon: HeartPulse, title: "rPPG signal", sub: "Extracting the pulse from skin-color changes" },
  { icon: Gauge, title: "Feature analysis", sub: "Measuring physiological features" },
  { icon: Atom, title: "Quantum classification", sub: "Hybrid quantum-classical scoring" },
  { icon: ShieldCheck, title: "Final decision", sub: "Issuing the verification verdict" },
];

/* backend reports 3 stages → map onto the 7-step pipeline */
export const stageActive = (idx) => (idx === 1 ? 3 : idx === 2 ? 5 : idx === 0 ? 0 : 7);

export const confWord = (c) =>
  c == null ? "No confidence value" : c >= 0.6 ? "High confidence" : c >= 0.3 ? "Moderate confidence" : "Low confidence";

export const fmtClock = (s) => {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
};

export const fmtVal = (v, digits = 3) => {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  if (typeof v !== "number") return String(v);
  return v.toFixed(digits);
};

export const fmtSize = (bytes) => {
  if (bytes == null) return "—";
  if (bytes >= 1048576) return `${(bytes / 1048576).toFixed(1)} MB`;
  return `${Math.max(1, Math.round(bytes / 1024))} KB`;
};

export const humanStatus = (lines = []) => {
  const last = [...lines]
    .reverse()
    .map((l) => l.replace(/^\[\d{2}:\d{2}:\d{2}\]\s*/, "").trim())
    .find((s) => s && !s.startsWith("[") && !s.startsWith("─"));
  return last || "Working…";
};

export const clamp01 = (v) => (v == null || Number.isNaN(v) ? 0 : Math.min(1, Math.max(0, v)));