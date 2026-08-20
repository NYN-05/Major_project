import { useEffect, useState } from "react";
import { motion, useReducedMotion } from "motion/react";
import { RefreshCw } from "lucide-react";
import { PLAIN, clamp01, TONE_ICON } from "../lib.js";

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
      <div className="conf-ring" role="img" aria-label={`P(live) ${Math.round(frac * 100)} percent`}>
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
          <span className="conf-pct mono">{value != null ? value.toFixed(3) : "—"}</span>
          <span className="conf-cap">P(live)</span>
        </div>
      </div>
    </div>
  );
}

export default function VerdictCard({ result, onTryAgain }) {
  const label = result?.verdict?.label ?? "UNCERTAIN";
  const p = PLAIN[label] ?? PLAIN.UNCERTAIN;
  const Icon = TONE_ICON[p.tone];
  const confidence = result?.verdict?.confidence;
  const probReal = result?.stages?.quantum?.prob_real;

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
          <div className="verdict-nums">
            <span className="verdict-num">
              <small>P(live)</small>
              <b className="mono">{probReal != null ? probReal.toFixed(3) : "—"}</b>
            </span>
          </div>
        </div>
        <ConfidenceRing value={probReal ?? confidence ?? 0.5} tone={p.tone} />
      </div>

      <div className="verdict-actions">
        <button type="button" className="btn btn-ghost" onClick={onTryAgain}>
          <RefreshCw size={13} aria-hidden="true" />
          Analyze another video
        </button>
      </div>
    </motion.section>
  );
}
