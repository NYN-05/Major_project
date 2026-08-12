import { motion } from "motion/react";
import { Check, Clock, Loader2 } from "lucide-react";
import { humanStatus, fmtClock, PIPELINE, stageActive } from "../lib.js";

const STAGE_NAMES = ["Preparing", "Frames", "Face detection", "rPPG signal", "Features", "Quantum", "Deciding"];

function Timeline({ active, done }) {
  return (
    <ol className="timeline" aria-label="verification pipeline (7 steps)">
      {PIPELINE.map(({ title, icon: Icon }, i) => {
        const state = done || i < active ? "done" : i === active ? "active" : "pending";
        return (
          <motion.li
            key={title}
            className={`ts ${state}`}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.05 + i * 0.05, duration: 0.35, ease: "easeOut" }}
          >
            <span className="ts-dot" aria-hidden="true">
              {state === "done" ? (
                <Check size={13} />
              ) : state === "active" ? (
                <motion.span
                  animate={{ rotate: 360 }}
                  transition={{ duration: 1.1, repeat: Infinity, ease: "linear" }}
                  style={{ display: "grid", placeItems: "center" }}
                >
                  <Loader2 size={13} />
                </motion.span>
              ) : (
                <Icon size={13} />
              )}
            </span>
            <p className="ts-title">{title}</p>
            <p className="ts-sub">{i <= active ? "…" : "waiting"}</p>
          </motion.li>
        );
      })}
    </ol>
  );
}

export default function ProcessingView({ stageIdx, elapsed, videoName, lines }) {
  const active = Math.min(stageActive(stageIdx), 6);
  const status = humanStatus(lines);
  const target = stageIdx === 0 ? 0 : stageIdx / 3;
  const pct = Math.round(target * 100);
  const lastLines = lines.slice(-4);

  return (
    <motion.section
      className="card processing"
      role="status"
      aria-live="polite"
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
    >
      <header className="processing-head">
        <p className="run-title">
          <span className="live-dot" aria-hidden="true" />
          Verification in progress
        </p>
        <div className="run-meta">
          <strong>{videoName}</strong>
          <span className="run-clock mono">
            <Clock size={11} /> {fmtClock(elapsed)}
          </span>
        </div>
      </header>

      <Timeline active={active} done={false} />

      <div className="progress-block">
        <div className="pb-row">
          <span className="pb-status">
            Stage {active + 1} of 7 — <strong>{STAGE_NAMES[active]}</strong> · {status}
          </span>
          <span className="pb-percent mono">{pct}%</span>
        </div>
        <div
          className="pb-track"
          role="progressbar"
          aria-label="verification progress"
          aria-valuenow={pct}
          aria-valuemin={0}
          aria-valuemax={100}
        >
          <motion.span
            className={`pb-fill${stageIdx === 0 ? " indet" : ""}`}
            initial={false}
            animate={{ scaleX: stageIdx === 0 ? undefined : pct / 100 }}
            style={stageIdx === 0 ? undefined : { transform: `scaleX(${pct / 100})` }}
          />
        </div>
      </div>

      <div className="log-panel">
        <div className="log-head">Live pipeline log</div>
        {lastLines.length ? (
          <ol className="log-lines mono" aria-live="polite">
            {lastLines.map((l, i) => (
              <li key={`${l}-${i}`}>{l.replace(/^\[\d{2}:\d{2}:\d{2}\]\s*/, "")}</li>
            ))}
          </ol>
        ) : (
          <p className="log-empty mono">waiting for pipeline output…</p>
        )}
      </div>
    </motion.section>
  );
}