import { motion } from "motion/react";
import { AlertTriangle, CheckCircle2, RefreshCw, XCircle } from "lucide-react";
import { PLAIN, clamp01, confWord, fmtVal } from "../lib.js";

const STATE_ICON = {
  ok: CheckCircle2,
  bad: XCircle,
  warn: AlertTriangle,
};

function RadialGauge({ value, tone, size = 132 }) {
  const r = (size - 14) / 2;
  const c = 2 * Math.PI * r;
  const frac = clamp01(value);
  return (
    <div className={`gauge gauge-${tone}`} style={{ width: size, height: size }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} role="img" aria-label={`confidence ${Math.round(frac * 100)} percent`}>
        <circle className="gauge-track" cx={size / 2} cy={size / 2} r={r} fill="none" strokeWidth="9" strokeDasharray={c} />
        <motion.circle
          className="gauge-arc"
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          strokeWidth="9"
          strokeLinecap="round"
          strokeDasharray={c}
          initial={{ strokeDashoffset: c }}
          animate={{ strokeDashoffset: c * (1 - frac) }}
          transition={{ duration: 1.2, ease: [0.22, 1, 0.36, 1], delay: 0.15 }}
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
        />
      </svg>
      <div className="gauge-center">
        <motion.span
          className="gauge-pct mono"
          initial={{ opacity: 0, scale: 0.7 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.5, duration: 0.4 }}
        >
          {Math.round(frac * 100)}%
        </motion.span>
        <span className="gauge-cap">confidence</span>
      </div>
    </div>
  );
}

export default function VerdictPanel({ result, onTryAgain }) {
  const label = result?.verdict?.label ?? "UNCERTAIN";
  const p = PLAIN[label] ?? PLAIN.UNCERTAIN;
  const Icon = STATE_ICON[p.tone];
  const confidence = result?.verdict?.confidence;
  const probReal = result?.stages?.quantum?.prob_real;
  const xc = result?.stages?.rppg_crosscheck;
  const reason = result?.verdict?.reason;

  return (
    <motion.section
      className={`verdict verdict-${p.tone}`}
      aria-label="verification result"
      initial={{ opacity: 0, y: 22, scale: 0.985 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
    >
      <div className="verdict-glow" aria-hidden="true" />

      <div className="verdict-gauge-col">
        <RadialGauge value={confidence ?? probReal ?? 0.5} tone={p.tone} />
        <span className="verdict-badge" aria-hidden="true">
          <Icon size={16} />
        </span>
      </div>

      <div className="verdict-body">
        <span className="sv-eyebrow">Verification result</span>
        <motion.h2
          className={`verdict-word tone-${p.tone}`}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.15, duration: 0.45 }}
        >
          {p.word}
        </motion.h2>
        <p className="verdict-note">{reason || p.note}</p>
        {confidence != null && (
          <p className="verdict-conf">
            {confWord(confidence)}
            {p.tone === "warn" && <span className="verdict-review">· recommended for manual review</span>}
          </p>
        )}

        <div className="verdict-metrics">
          <div className="vm">
            <span className="vm-k">Live-signal score</span>
            <span className="vm-v mono">{fmtVal(probReal, 3)}</span>
          </div>
          <div className="vm">
            <span className="vm-k">Model confidence</span>
            <span className="vm-v mono">{fmtVal(confidence, 3)}</span>
          </div>
          {xc?.verdict && (
            <div className="vm">
              <span className="vm-k">ML cross-check</span>
              <span className={`vm-v mono${xc.verdict === "DEEPFAKE" ? " text-bad" : " text-ok"}`}>
                {xc.verdict === "DEEPFAKE" ? "deepfake" : "real"}
              </span>
            </div>
          )}
        </div>

        <div className="sv-actions">
          <motion.button
            type="button"
            className="btn btn-primary"
            onClick={onTryAgain}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.97 }}
          >
            <RefreshCw size={14} aria-hidden="true" />
            Analyze another video
          </motion.button>
        </div>
      </div>
    </motion.section>
  );
}