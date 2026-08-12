import { useEffect, useState } from "react";
import { motion, useReducedMotion } from "motion/react";
import { RefreshCw } from "lucide-react";
import { PLAIN, probabilityWord, clamp01, TONE_ICON, GLOSSARY } from "../lib.js";
import Tip from "./Tip.jsx";

function ConfidenceRing({ value, tone }) {
  const reduced = useReducedMotion();
  const [display, setDisplay] = useState(0);
  const frac = clamp01(value);

  useEffect(() => {
    if (reduced) {
      setDisplay(frac);
      return undefined;
    }
    let raf;
    const t0 = performance.now();
    const dur = 900;
    const step = (t) => {
      const e = 1 - Math.pow(1 - Math.min(1, (t - t0) / dur), 3);
      setDisplay(e * frac);
      if (e < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [frac, reduced]);

  const size = 104;
  const r = (size - 12) / 2;
  const c = 2 * Math.PI * r;

  return (
    <div className="conf-wrap">
      <div className="conf-ring" role="img" aria-label={`confidence ${Math.round(frac * 100)} percent`}>
        <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
          <circle className="conf-track" cx={size / 2} cy={size / 2} r={r} fill="none" strokeWidth="8" strokeDasharray={c} />
          <circle
            className="conf-arc"
            cx={size / 2}
            cy={size / 2}
            r={r}
            fill="none"
            strokeWidth="8"
            strokeDasharray={c}
            strokeDashoffset={c * (1 - display)}
          />
        </svg>
        <div className="conf-center">
          <span className="conf-pct mono">{Math.round(frac * 100)}%</span>
          <span className="conf-cap">confidence</span>
        </div>
      </div>
      <span className="conf-note">confidence of the final quantum verdict</span>
    </div>
  );
}

export default function VerdictCard({ result, onTryAgain }) {
  const label = result?.verdict?.label ?? "UNCERTAIN";
  const p = PLAIN[label] ?? PLAIN.UNCERTAIN;
  const Icon = TONE_ICON[p.tone];
  const confidence = result?.verdict?.confidence;
  const probReal = result?.stages?.quantum?.prob_real;
  const reason = result?.verdict?.reason;

  return (
    <motion.section
      className={`card verdict tone-${p.tone}`}
      aria-label="verification verdict"
      initial={{ opacity: 0, y: 18 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.55, ease: [0.22, 1, 0.36, 1], delay: 0.05 }}
    >
      <div className="verdict-top">
        <div>
          <span className="verdict-eyebrow">
            <Icon size={13} aria-hidden="true" /> Final verdict
          </span>
          <h3 className="verdict-word">{p.word}</h3>
          <p className="verdict-cause">
            {reason || p.note} —{" "}
            <Tip text={GLOSSARY.probability}>
              probability {probReal == null ? "" : `${(probReal * 100).toFixed(0)}% `}
            </Tip>
            of live by the quantum layer.
          </p>
        </div>
        <ConfidenceRing value={confidence ?? probReal ?? 0.5} tone={p.tone} />
      </div>

      <div className="verdict-prob">
        <div className="vp-row">
          <span className="vp-k">
            <Tip text={GLOSSARY.probability}>Probability of live</Tip>
          </span>
          <span className="vp-track" aria-hidden="true">
            <span className="vp-fill" style={{ transform: `scaleX(${clamp01(probReal)})` }} />
          </span>
          <span className="vp-v mono">{probReal != null ? probReal.toFixed(3) : "—"}</span>
        </div>
        <div className="vp-row" style={{ gridTemplateColumns: "118px 1fr" }}>
          <span className="vp-k">
            <Tip text={GLOSSARY.confidence}>Confidence</Tip>
          </span>
          <span className="vp-v" style={{ textAlign: "left", fontSize: "11px", color: "var(--dim)" }}>
            {probabilityWord(probReal)}
            {confidence != null && ` · decisiveness ${confidence.toFixed(3)}`}
          </span>
        </div>
      </div>

      {reason ? (
        <p className="verdict-reason">{reason}</p>
      ) : (
        <p className="verdict-reason" style={{ borderLeftColor: "transparent", background: "transparent", padding: 0, marginTop: 6 }}>
          <span style={{ color: "var(--dim)" }}>{p.note}</span>
        </p>
      )}

      <div className="verdict-actions">
        <button type="button" className="btn btn-ghost" onClick={onTryAgain}>
          <RefreshCw size={13} aria-hidden="true" />
          Analyze another video
        </button>
      </div>
    </motion.section>
  );
}