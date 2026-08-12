import { motion } from "motion/react";
import { Activity, BrainCircuit, HeartPulse, ScanSearch } from "lucide-react";
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
  const frames = result?.stages?.frames?.stats ?? null;
  const rppg = result?.stages?.rppg ?? null;
  const xc = result?.stages?.rppg_crosscheck ?? null;

  const accepted = frames?.accepted_frames ?? null;
  const sampled = frames?.sampled_frames ?? null;
  const usable = rppg?.n_frames_usable ?? null;
  const total = rppg?.n_frames_total ?? null;
  const inputMode = rppg?.input_mode ?? null;

  const bpm = rppg?.features?.heart_rate_bpm ?? null;
  const sqi = rppg?.features?.signal_quality_index ?? null;
  const sqiTone = sqi == null ? "" : sqi >= 0.6 ? "" : "sqi-warn";

  const framesSub =
    usable != null && total != null
      ? `${usable} of ${total} frames used by rPPG`
      : "no frame data";

  const framesSub2 = inputMode
    ? inputMode === "stage1_frames"
      ? "stage-1 accepted frames + face re-tracking"
      : accepted != null && sampled != null
        ? `${accepted}/${sampled} passed stage-1 quality gate; full video read fallback`
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
        transition={{ duration: 0.4, ease: "easeOut", delay: 0.2 }}
      >
        <span className="metric-k">
          <ScanSearch size={13} aria-hidden="true" />
          Usable / accepted frames
        </span>
        <span className="metric-v">
          {usable != null ? usable : "—"}
          <small> / {total ?? "—"}</small>
        </span>
        <span className="metric-s">{framesSub}</span>
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