import {
  Atom,
  CheckCircle2,
  Clapperboard,
  Gauge,
  HeartPulse,
  ScanFace,
  ScanSearch,
  ShieldAlert,
  ShieldCheck,
  UserCheck,
} from "lucide-react";

export const PLAIN = {
  REAL: {
    word: "Likely Real",
    tone: "ok",
    verdictLabel: "LIKELY REAL",
    headline: "Verified as a live person",
    note: "The facial pulse extracted from this video matches the physiological signature of a real, live person. The hybrid quantum-classical classifier found no evidence of synthesis or manipulation.",
  },
  FAKE: {
    word: "Likely Deepfake",
    tone: "bad",
    verdictLabel: "LIKELY DEEPFAKE",
    headline: "Synthetic or manipulated footage",
    note: "This video does not show the natural pulse of a live person. The physiological signal is inconsistent with a genuine recording — consistent with AI-generated or manipulated footage.",
  },
  UNCERTAIN: {
    word: "Needs Human Review",
    tone: "warn",
    verdictLabel: "NEEDS HUMAN REVIEW",
    headline: "Signal too weak for an automated decision",
    note: "The extracted physiological signal did not reach the confidence required for an automated verdict. This video should be passed to a manual KYC reviewer before use.",
  },
};

export const PIPELINE = [
  { title: "Video received", sub: "Reading the uploaded clip", icon: Clapperboard },
  { title: "Frame quality", sub: "Filtering blurry or dark frames", icon: ScanSearch },
  { title: "Face detection", sub: "Locating the face region", icon: ScanFace },
  { title: "rPPG signal", sub: "Extracting the pulse from skin-color changes", icon: HeartPulse },
  { title: "Feature analysis", sub: "Measuring 10 physiological features", icon: Gauge },
  { title: "Quantum classifier", sub: "QAOA selection + hybrid VQC scoring", icon: Atom },
  { title: "Final decision", sub: "Issuing the verification verdict", icon: ShieldCheck },
];

/* backend reports 3 run stages → map onto the 7-step pipeline */
export const stageActive = (idx) => (idx === 1 ? 3 : idx === 2 ? 5 : idx === 0 ? 0 : 7);

export const confWord = (c) =>
  c == null ? "No confidence value" : c >= 0.6 ? "High confidence" : c >= 0.3 ? "Moderate confidence" : "Low confidence";

export const probabilityWord = (p) =>
  p == null
    ? "No probability value"
    : p >= 0.75
      ? "Strongly favors real"
      : p >= 0.55
        ? "Leans toward real"
        : p >= 0.45
          ? "Ambiguous signal"
          : p >= 0.25
            ? "Leans toward fake"
            : "Strongly favors fake";

export const fmtClock = (s) => {
  if (s == null || !Number.isFinite(s)) return "—";
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

export const clamp01 = (v) => (v == null || Number.isNaN(v) ? 0 : Math.min(1, Math.max(0, v)));

export const humanStatus = (lines = []) => {
  const last = [...lines]
    .reverse()
    .map((l) => l.replace(/^\[\d{2}:\d{2}:\d{2}\]\s*/, "").trim())
    .find((s) => s && !s.startsWith("[") && !s.startsWith("─"));
  return last || "Working…";
};

export const fmtTimestamp = (iso) => {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return String(iso);
  return d.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
};

/* ------------------------------------------------------------------ */
/* glossary — tooltips for technical terms                             */
/* ------------------------------------------------------------------ */

export const GLOSSARY = {
  rppg: "Remote photoplethysmography — measuring the pulse from subtle skin-color changes in video, without skin contact.",
  sqi: "Signal Quality Index — how clean and periodic the extracted pulse is (0–1); higher is more physiological.",
  qaoa: "Quantum Approximate Optimization Algorithm — used on this pipeline to select the 6 most informative features from the 10 extracted ones.",
  vqc: "Variational Quantum Circuit — a parameterized quantum circuit (PennyLane) whose weights are trained classically; the hybrid decision layer.",
  probability: "Probability of live — the model's estimate that the recording comes from a real, living person (0–1).",
  confidence: "Confidence — the model's estimate that the recording is live (0–1), identical to the probability of live.",
  snr: "Signal-to-noise ratio of the extracted pulse in decibels; negative values mean the pulse is buried in noise.",
  prv: "Pulse rate variability — beat-to-beat variation in milliseconds; real physiology fluctuates naturally.",
  entropy: "Spectral entropy of the pulse spectrum; low entropy = clean periodic pulse, high entropy = noisy or irregular.",
  mad: "Mean absolute deviation of the pulse waveform — how much the signal amplitude varies around its average.",
  hrd: "HR half-diff — the absolute difference between the heart rate estimated from the first and second half of the clip; physiological recordings stay consistent across the clip.",
  ppr: "Spectral peak prominence — how much the dominant in-band pulse peak stands above the mean spectrum; a strong clean pulse has a high peak-to-mean ratio.",
  roi: "Region of Interest — the skin patches (left cheek, right cheek, forehead) the pulse is measured from.",
  crosscheck: "A separate, simpler Random-Forest model trained directly on the same rPPG feature vector. It is informational only — it never joins the quantum decision.",
};

/* ------------------------------------------------------------------ */
/* rPPG feature labels (order matches RPPGFeatures.feature_names)      */
/* ------------------------------------------------------------------ */

export const FEATURE_LABELS = [
  { key: "heart_rate_bpm", label: "Estimated heart rate", unit: "BPM", tip: GLOSSARY.rppg },
  { key: "snr_db", label: "Signal-to-noise ratio", unit: "dB", tip: GLOSSARY.snr },
  { key: "prv_std_ms", label: "Pulse-rate variability", unit: "ms", tip: GLOSSARY.prv },
  { key: "spectral_entropy", label: "Spectral entropy", unit: "", tip: GLOSSARY.entropy },
  { key: "mad", label: "Mean absolute deviation", unit: "", tip: GLOSSARY.mad },
  { key: "signal_quality_index", label: "Signal quality index", unit: "0–1", tip: GLOSSARY.sqi },
  { key: "cheek_forehead_correlation", label: "Cheek–forehead correlation", unit: "", tip: GLOSSARY.roi },
  { key: "left_right_cheek_correlation", label: "Left–right cheek correlation", unit: "", tip: GLOSSARY.roi },
  { key: "hr_half_diff", label: "HR half-diff", unit: "BPM", tip: GLOSSARY.hrd },
  { key: "peak_prominence", label: "Peak prominence", unit: "", tip: GLOSSARY.ppr },
];

export const featureInfo = (key) => FEATURE_LABELS.find((f) => f.key === key);

/* ROI sources used by the backend when blending the pulse (RPPGPipeline defaults) */
export const ROIS = [
  { name: "Left cheek", weight: 35, tone: "roi-lc" },
  { name: "Right cheek", weight: 35, tone: "roi-rc" },
  { name: "Forehead", weight: 30, tone: "roi-fh" },
];

/* ------------------------------------------------------------------ */
/* icons mapped to verdict tones                                       */
/* ------------------------------------------------------------------ */

export const TONE_ICON = {
  ok: CheckCircle2,
  bad: ShieldAlert,
  warn: ShieldAlert,
};

export const VERDICT_ICON = {
  REAL: UserCheck,
  FAKE: ShieldAlert,
  UNCERTAIN: ShieldAlert,
};