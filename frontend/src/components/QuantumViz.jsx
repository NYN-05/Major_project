import { motion } from "motion/react";
import { Atom, Check, CircleDot, GitBranch } from "lucide-react";

const FLOW = [
  { title: "Features", sub: "8 physiological inputs from the pulse" },
  { title: "Feature selection", sub: "QAOA picks the strongest features" },
  { title: "Quantum encoding", sub: "features mapped onto qubit rotation angles" },
  { title: "Hybrid VQC", sub: "PennyLane circuit + trainable weights" },
  { title: "Score", sub: "final probability that the video is live" },
];

export default function QuantumViz({ result }) {
  const quantum = result?.stages?.quantum ?? null;
  const probReal = quantum?.prob_real ?? null;
  const confidence = quantum?.confidence ?? null;
  const selected = Array.isArray(quantum?.selected_features) ? quantum.selected_features : null;
  const verdictLabel = quantum?.verdict ?? null;

  if (!quantum) return null;

  return (
    <motion.section
      className="quant card-soft"
      initial={{ opacity: 0, y: 18 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: "easeOut", delay: 0.2 }}
    >
      <div className="section-head">
        <span className="section-ico" aria-hidden="true">
          <Atom size={15} />
        </span>
        <h3 className="section-title">Quantum classification flow</h3>
        <span className="section-rule" aria-hidden="true" />
        <span className="chip chip-quant mono">PennyLane · hybrid VQC</span>
      </div>

      <div className="quant-flow" role="list">
        {FLOW.map((step, i) => {
          const done = i < FLOW.length - 1;
          return (
            <div className="qstep" role="listitem" key={step.title}>
              <div className="qnode">
                {i === 0 ? (
                  <CircleDot size={13} />
                ) : i === 1 ? (
                  <GitBranch size={13} />
                ) : done ? (
                  <Check size={13} />
                ) : (
                  <Atom size={13} />
                )}
              </div>
              <p className="qstep-title">{step.title}</p>
              <p className="qstep-sub">{step.sub}</p>
              {i < FLOW.length - 1 && <span className="qconn" aria-hidden="true" />}
            </div>
          );
        })}
      </div>

      <div className="quant-grid">
        {probReal != null && (
          <div className="qg">
            <span className="qg-k">Probability of live</span>
            <span className="qg-v mono">{probReal.toFixed(3)}</span>
          </div>
        )}
        {confidence != null && (
          <div className="qg">
            <span className="qg-k">Quantum confidence</span>
            <span className="qg-v mono">{confidence.toFixed(3)}</span>
          </div>
        )}
        {verdictLabel && (
          <div className="qg">
            <span className="qg-k">Quantum verdict</span>
            <span className="qg-v mono">{verdictLabel}</span>
          </div>
        )}
        {selected && (
          <div className="qg qg-wide">
            <span className="qg-k">Features selected by QAOA</span>
            <span className="qg-tags mono">{selected.join(" · ")}</span>
          </div>
        )}
      </div>
    </motion.section>
  );
}