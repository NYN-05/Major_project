import { motion } from "motion/react";
import { Activity, BrainCircuit, HeartPulse } from "lucide-react";
import { fmtVal, clamp01, GLOSSARY } from "../lib.js";
import Tip from "./Tip.jsx";

function Bar({ pct, tone = "" }) {
  return (
    <div className={`metric-bar${tone ? ` ${tone}` : ""}`} aria-hidden="true">
      <i style={{ transform: `scaleX(${clamp01(pct)})` }} />
    </div>
  );
}

export default function KeyMetrics({ result }) {
  const rppg = result?.stages?.rppg ?? null;
  const xc = result?.stages?.rppg_crosscheck ?? null;

  const inputMode = rppg?.input_mode ?? null;

  const bpm = rppg?.features?.heart_rate_bpm ?? null;
  const sqi = rppg?.features?.signal_quality_index ?? null;
  const sqiTone = sqi == null ? "" : sqi >= 0.6 ? "" : "sqi-warn";

  const framesSub2 = inputMode
    ? inputMode === "stage1_frames"
      ? "stage-1 accepted frames + face re-tracking"
      : "full video read fallback"
    : "";

  return (
    <section className="metrics-grid" aria-label="key metrics">
      <motion.div
        className="metric"
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: "easeOut", delay: 0.1 }}
      >
        <span className="metric-k">
          <HeartPulse size={13} aria-hidden="true" />
          <Tip text={GLOSSARY.rppg}>Estimated heart rate</Tip>
        </span>
        <span className="metric-v">
          {bpm != null ? <>{Math.round(bpm)}</> : "—"}
          <small> BPM</small>
        </span>
        <span className="metric-s">
          from the extracted pulse · {framesSub2 || "position-normalized"}
        </span>
      </motion.div>

      <motion.div
        className="metric"
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: "easeOut", delay: 0.15 }}
      >
        <span className="metric-k">
          <Activity size={13} aria-hidden="true" />
          <Tip text={GLOSSARY.sqi}>rPPG signal quality</Tip>
        </span>
        <span className="metric-v">{sqi != null ? <>{Math.round(sqi * 100)}<small>%</small></> : "—"}</span>
        <span className="metric-s">signal quality index of the reconstructed pulse</span>
        <Bar pct={sqi} tone={sqiTone} />
      </motion.div>

      <motion.div
        className="metric"
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: "easeOut", delay: 0.25 }}
      >
        <span className="metric-k">
          <BrainCircuit size={13} aria-hidden="true" />
          <Tip text={GLOSSARY.crosscheck}>Secondary ML cross-check</Tip>
        </span>
        {xc?.verdict ? (
          <>
            <span className="metric-v">
              <span className={`metric-value-${xc.verdict === "DEEPFAKE" ? "bad" : "ok"}`}>
                {xc.verdict === "DEEPFAKE" ? "Deepfake" : "Real"}
              </span>
            </span>
            <span className="metric-s">Random Forest on the same rPPG features · informational only</span>
          </>
        ) : (
          <>
            <span className="metric-v">—</span>
            <span className="metric-s">cross-check not available for this run</span>
          </>
        )}
      </motion.div>
    </section>
  );
}