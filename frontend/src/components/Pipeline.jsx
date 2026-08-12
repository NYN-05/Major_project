import { useEffect, useState } from "react";
import { motion } from "motion/react";
import {
  Atom,
  CheckCircle2,
  Clapperboard,
  Gauge,
  HeartPulse,
  Loader2,
  ScanFace,
  ScanSearch,
  ShieldCheck,
} from "lucide-react";
import ProgressBar from "./ProgressBar.jsx";
import { PLAIN, fmtClock, humanStatus, stageActive } from "../lib.js";

const STAGE_NAMES = ["Preparing", "Frames", "Face detection", "rPPG signal", "Features", "Quantum", "Deciding"];

const PIPELINE_STEPS = [
  { icon: Clapperboard, title: "Video received" },
  { icon: ScanSearch, title: "Frame quality" },
  { icon: ScanFace, title: "Face detection" },
  { icon: HeartPulse, title: "rPPG signal" },
  { icon: Gauge, title: "Feature analysis" },
  { icon: Atom, title: "Quantum classification" },
  { icon: ShieldCheck, title: "Final decision" },
];

const PIPELINE_SUBS = [
  "Reading the uploaded clip",
  "Filtering blurry or dark frames",
  "Locating the face region",
  "Extracting the pulse from skin-color changes",
  "Measuring physiological features",
  "Hybrid quantum-classical scoring",
  "Issuing the verification verdict",
];

export default function Pipeline({ stageIdx, lines, elapsed, videoName, phase, result, resultElapsed }) {
  const active = Math.min(stageActive(stageIdx), 6);
  const status = humanStatus(lines);
  const done = phase === "done";

  /* creep toward the stage target so the bar is always visibly moving */
  const target = stageIdx === 0 ? null : stageIdx / 3;
  const [value, setValue] = useState(target);
  useEffect(() => {
    if (done) {
      setValue(1);
      return undefined;
    }
    if (target === null) {
      setValue(null);
      return undefined;
    }
    let raf;
    const tick = () => {
      setValue((v) => {
        if (v === null) return target;
        const next = v + (target - v) * 0.015 + 0.0015;
        return next >= target ? target : next;
      });
    };
    const id = setInterval(tick, 120);
    return () => clearInterval(id);
  }, [target, done]);

  const label = result?.verdict?.label ?? "UNCERTAIN";
  const p = PLAIN[label] ?? PLAIN.UNCERTAIN;
  const confidence = result?.verdict?.confidence ?? result?.stages?.quantum?.confidence;
  const lastLines = lines.slice(-3);

  return (
    <motion.section
      className="pipeline"
      role="status"
      aria-live="polite"
      initial={{ opacity: 0, y: 18 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.55, ease: [0.22, 1, 0.36, 1] }}
    >
      <header className="pipeline-head">
        <div className="run-head">
          <span className={`live-dot${done ? " live-dot-done" : ""}`} aria-hidden="true" />
          <span className="run-title">{done ? "Verification complete" : "Verification pipeline"}</span>
        </div>
        <div className="pipeline-meta">
          {videoName && <span className="run-file">{videoName}</span>}
          <span className="run-clock mono">{fmtClock(done ? (resultElapsed ?? 0) : elapsed)}</span>
        </div>
      </header>

      <ProgressBar
        value={done ? 100 : value}
        pendingLabel="Working"
        completeLabel="Analysis complete"
        ariaLabel={videoName ? `Checking ${videoName}` : "Analysis progress"}
        className="pb-inline"
      />

      {/* live / final result panel — directly below the progress bar */}
      <div className={`pl-panel${done ? " pl-done" : ""}`}>
        {done ? (
          <>
            <span className="pl-ico" aria-hidden="true">
              <CheckCircle2 size={16} />
            </span>
            <div className="pl-body">
              <p className="pl-eyebrow">Result</p>
              <p className={`pl-word tone-${p.tone}`}>{p.word}</p>
            </div>
            <div className="pl-stats mono">
              <span className="pl-stat">
                <span className="pl-stat-k">confidence</span>
                <span className="pl-stat-v">{confidence != null ? `${Math.round(confidence * 100)}%` : "—"}</span>
              </span>
              <span className="pl-stat">
                <span className="pl-stat-k">runtime</span>
                <span className="pl-stat-v">{fmtClock(resultElapsed ?? 0)}</span>
              </span>
            </div>
          </>
        ) : (
          <>
            <span className="pl-ico" aria-hidden="true">
              <Loader2 size={14} className="pipe-spin" />
            </span>
            <div className="pl-body">
              <p className="pl-eyebrow">
                Stage {active + 1} of 7 · {STAGE_NAMES[active]}
              </p>
              <p className="pl-status">{status}</p>
            </div>
            <ol className="pl-lines mono">
              {lastLines.map((l, i) => (
                <li key={`${l}-${i}`}>{l.replace(/^\[\d{2}:\d{2}:\d{2}\]\s*/, "")}</li>
              ))}
            </ol>
          </>
        )}
      </div>

      <ol className="pipe">
        {PIPELINE_STEPS.map(({ icon: Icon, title }, i) => {
          const state = done ? "done" : i < active ? "done" : i === active ? "active" : "pending";
          return (
            <motion.li
              key={title}
              className={`pipe-step ${state}`}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 + i * 0.07, duration: 0.4, ease: "easeOut" }}
            >
              <span className="pipe-rail" aria-hidden="true" />
              <span className="pipe-dot" aria-hidden="true">
                <Icon size={15} />
              </span>
              <div className="pipe-body">
                <p className="pipe-title">{title}</p>
                <p className="pipe-sub">
                  {done ? "complete" : state === "active" ? status : PIPELINE_SUBS[i]}
                </p>
              </div>
              <span className="pipe-state" aria-label={state}>
                {state === "done" && <CheckCircle2 size={14} />}
                {state === "active" && (
                  <motion.span
                    className="pipe-spin"
                    animate={{ rotate: 360 }}
                    transition={{ duration: 1.1, repeat: Infinity, ease: "linear" }}
                  >
                    <Loader2 size={13} />
                  </motion.span>
                )}
                {state === "pending" && (
                  <span className="pipe-num mono">{String(i + 1).padStart(2, "0")}</span>
                )}
              </span>
            </motion.li>
          );
        })}
      </ol>
    </motion.section>
  );
}