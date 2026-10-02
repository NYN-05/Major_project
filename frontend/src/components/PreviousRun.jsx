import { History } from "lucide-react";
import { PLAIN, TONE_ICON, fmtTimestamp } from "../lib.js";

export default function PreviousRun({ previous, onOpen }) {
  if (!previous?.video || !previous?.verdict?.label) return null;

  const plain = PLAIN[previous.verdict.label] ?? PLAIN.FAKE;
  const Icon = TONE_ICON[plain.tone];
  const prob = previous?.stages?.quantum?.prob_real ?? previous.verdict.confidence;

  return (
    <section className="card card-pad previous" aria-label="Previous verification">
      <div className="section-head">
        <span className="section-ico" aria-hidden="true">
          <History size={13} />
        </span>
        <div>
          <h2 className="section-title">Last verification</h2>
          <p className="section-sub">The most recent result on this machine</p>
        </div>
      </div>

      <button type="button" className="vp-row" onClick={onOpen}>
        <span className={`vp-ico tone-${plain.tone}`} aria-hidden="true">
          <Icon size={15} />
        </span>
        <span className="vp-body">
          <span className="vp-name">{previous.video}</span>
          <span className="vp-meta">
            {fmtTimestamp(previous.timestamp)}
            <span className="sep" aria-hidden="true" />
            {plain.verdictLabel}
          </span>
        </span>
        <span className="vp-right">
          <span className={`vp-word text-${plain.tone}`}>{plain.word}</span>
          <span className="vp-conf mono">
            P(live) {prob != null && Number.isFinite(prob) ? prob.toFixed(3) : "—"}
          </span>
        </span>
      </button>
    </section>
  );
}