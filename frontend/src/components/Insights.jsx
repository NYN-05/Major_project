import { useEffect, useRef, useState } from "react";
import { animate, motion } from "motion/react";
import {
  Activity,
  Atom,
  BrainCircuit,
  HeartPulse,
  ScanSearch,
  ShieldCheck,
} from "lucide-react";
import { clamp01, fmtVal } from "../lib.js";

function CountUp({ value, digits = 0, className = "" }) {
  const [display, setDisplay] = useState(0);
  const prevRef = useRef(0);
  useEffect(() => {
    const target = Number.isFinite(value) ? value : 0;
    const from = prevRef.current;
    if (target === from) {
      setDisplay(target);
      return undefined;
    }
    const controls = animate(from, target, {
      duration: 0.9,
      ease: [0.22, 1, 0.36, 1],
      onUpdate: (v) => setDisplay(v),
    });
    prevRef.current = target;
    return () => controls.stop();
  }, [value]);
  return <span className={className}>{display.toFixed(digits)}</span>;
}

function Ring({ pct, tone = "" }) {
  const r = 15;
  const c = 2 * Math.PI * r;
  return (
    <svg className={`mring mring-${tone}`} width="38" height="38" viewBox="0 0 38 38" role="img" aria-label={`${Math.round(pct * 100)} percent`}>
      <circle className="mring-track" cx="19" cy="19" r={r} fill="none" strokeWidth="4" strokeDasharray={c} />
      <motion.circle
        className="mring-arc"
        cx="19"
        cy="19"
        r={r}
        fill="none"
        strokeWidth="4"
        strokeLinecap="round"
        strokeDasharray={c}
        initial={{ strokeDashoffset: c }}
        animate={{ strokeDashoffset: c * (1 - clamp01(pct)) }}
        transition={{ duration: 1, ease: [0.22, 1, 0.36, 1] }}
        transform="rotate(-90 19 19)"
      />
    </svg>
  );
}

function Bar({ pct, tone = "", min = 0, max = 1 }) {
  return (
    <div className={`mbar mbar-${tone}`} role="progressbar" aria-valuenow={Math.round(clamp01(pct) * 100)} aria-valuemin={min} aria-valuemax={max} aria-label="value">
      <motion.div
        className="mbar-fill"
        initial={{ scaleX: 0 }}
        animate={{ scaleX: clamp01(pct) }}
        transition={{ duration: 0.9, ease: [0.22, 1, 0.36, 1] }}
      />
    </div>
  );
}

function Metric({ icon: Icon, label, value, sub, render }) {
  return (
    <motion.div
      className="metric"
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, ease: "easeOut" }}
    >
      <div className="metric-top">
        <span className="metric-ico" aria-hidden="true">
          <Icon size={15} />
        </span>
        <span className="metric-label">{label}</span>
      </div>
      <div className="metric-value">{value}</div>
      <div className="metric-sub">{sub}</div>
      {render}
    </motion.div>
  );
}

export default function Insights({ result }) {
  const frames = result?.stages?.frames?.stats ?? null;
  const rppg = result?.stages?.rppg ?? null;
  const quantum = result?.stages?.quantum ?? null;
  const xc = result?.stages?.rppg_crosscheck ?? null;

  const accepted = frames?.accepted_frames ?? 0;
  const sampled = frames?.sampled_frames ?? 0;
  const acceptedPct = sampled > 0 ? accepted / sampled : 0;
  const bpm = rppg?.features?.heart_rate_bpm ?? null;
  const sqi = rppg?.features?.signal_quality_index ?? null;
  const prv = rppg?.features?.prv_std_ms ?? null;
  const probReal = quantum?.prob_real ?? null;
  const quantConf = quantum?.confidence ?? null;
  const featCount = Array.isArray(quantum?.selected_features) ? quantum.selected_features.length : null;

  return (
    <motion.section
      className="insights"
      initial={{ opacity: 0, y: 18 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: "easeOut", delay: 0.12 }}
    >
      <div className="section-head">
        <span className="section-ico" aria-hidden="true">
          <Activity size={15} />
        </span>
        <h3 className="section-title">Verification insights</h3>
        <span className="section-rule" aria-hidden="true" />
      </div>

      <div className="metrics">
        <Metric
          icon={ScanSearch}
          label="Frames analyzed"
          value={`${accepted} / ${sampled}`}
          sub={sampled > 0 ? "accepted after quality check" : "no frame data"}
          render={<Ring pct={acceptedPct} tone="ok" />}
        />
        <Metric
          icon={HeartPulse}
          label="Detected pulse"
          value={bpm != null ? <CountUp value={bpm} digits={0} className="metric-num" /> : "—"}
          sub={bpm != null ? `${rppg.n_frames_usable} usable frames` : "insufficient usable frames"}
        />
        <Metric
          icon={Activity}
          label="Signal quality"
          value={sqi != null ? <CountUp value={sqi * 100} digits={0} className="metric-num" /> : "—"}
          sub={sqi != null ? "rPPG signal quality index" : "not available"}
          render={sqi != null ? <Bar pct={sqi} tone={sqi >= 0.6 ? "ok" : "warn"} /> : null}
        />
        <Metric
          icon={Atom}
          label="Live-signal score"
          value={probReal != null ? <span className="metric-num">{fmtVal(probReal, 3)}</span> : "—"}
          sub={probReal != null ? "quantum classifier output" : "not available"}
          render={probReal != null ? <Bar pct={probReal} tone="lime" /> : null}
        />
        <Metric
          icon={BrainCircuit}
          label="ML cross-check"
          value={
            xc?.verdict ? (
              <span className={`metric-num${xc.verdict === "DEEPFAKE" ? " text-bad" : " text-ok"}`}>
                {fmtVal(xc.probability, 3)}
              </span>
            ) : (
              "—"
            )
          }
          sub={xc?.verdict ? `independent model: ${xc.verdict.toLowerCase()}` : "not available"}
          render={xc?.probability != null ? <Bar pct={xc.probability} tone={xc.verdict === "DEEPFAKE" ? "bad" : "ok"} /> : null}
        />
        <Metric
          icon={ShieldCheck}
          label="Quantum confidence"
          value={quantConf != null ? <span className="metric-num">{fmtVal(quantConf, 3)}</span> : "—"}
          sub={featCount != null ? `${featCount} features selected by QAOA` : "not available"}
          render={quantConf != null ? <Ring pct={quantConf} tone="lime" /> : null}
        />
      </div>
    </motion.section>
  );
}