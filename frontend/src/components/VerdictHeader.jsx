import { motion } from "motion/react";
import { CheckCircle2, ShieldCheck, ShieldAlert } from "lucide-react";
import { fmtClock, fmtTimestamp, PLAIN, PIPELINE } from "../lib.js";

const HEAD_ICON = {
  ok: ShieldCheck,
  bad: ShieldAlert,
  warn: ShieldAlert,
};

export default function VerdictHeader({ result, videoMeta, resultElapsed }) {
  const label = result?.verdict?.label ?? "UNCERTAIN";
  const p = PLAIN[label] ?? PLAIN.UNCERTAIN;
  const HeadIcon = HEAD_ICON[p.tone];
  const name = result?.video ?? "—";
  const ts = fmtTimestamp(result?.timestamp);
  const duration = videoMeta?.duration ?? result?.stages?.frames?.stats?.duration_s;

  return (
    <motion.section
      className="card verdict-header"
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
    >
      <div className="vh-left" style={{ flexGrow: 1 }}>
        <span className={`vh-icon${p.tone !== "ok" ? ` vh-ico-${p.tone}` : ""}`} aria-hidden="true">
          <HeadIcon size={18} />
        </span>
        <div>
          <h2 className="vh-title">{name}</h2>
          <div className="vh-meta">
            {duration != null && <span>{fmtClock(duration)} duration</span>}
            {duration != null && <span className="sep" aria-hidden="true" />}
            <span>Analyzed {ts}</span>
            {resultElapsed != null && (
              <>
                <span className="sep" aria-hidden="true" />
                <span>processed in {fmtClock(resultElapsed)}</span>
              </>
            )}
            <span className="sep" aria-hidden="true" />
            <span className="status-chip">
              <CheckCircle2 size={11} aria-hidden="true" /> Complete
            </span>
          </div>
        </div>
      </div>

      <ul
        className="timeline-compact"
        aria-label="verification timeline — all 7 steps complete"
      >
        {PIPELINE.map(({ title }) => (
          <li className="tc done" key={title}>
            <span className="tc-dot" aria-hidden="true">
              <i />
            </span>
            <label>{title}</label>
          </li>
        ))}
      </ul>
    </motion.section>
  );
}