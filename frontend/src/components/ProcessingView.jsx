import { motion } from "motion/react";
import { Check, Clock, Loader2 } from "lucide-react";
import { humanStatus, fmtClock, PIPELINE, stageActive } from "../lib.js";

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
            <p className="ts-sub">{i < active ? "done" : i === active ? "processing" : "waiting"}</p>
          </motion.li>
        );
      })}
    </ol>
  );
}

export default function ProcessingView({ stageIdx, elapsed, videoName, lines }) {
  const active = Math.min(stageActive(stageIdx), 6);
  const status = humanStatus(lines);
  // single source of truth: text, bar width and aria all read this value
  const pct = Math.round((active / 7) * 100);

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
            Stage {active + 1} of 7 — <strong>{PIPELINE[active].title}</strong> · {status}
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
            className="pb-fill"
            initial={false}
            animate={{ scaleX: pct / 100 }}
            transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
          />
        </div>
      </div>
    </motion.section>
  );
}