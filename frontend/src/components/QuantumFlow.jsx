import { motion } from "motion/react";
import { Atom, Check, CircleDot, GitBranch, Sigma, Sparkles } from "lucide-react";
import { GLOSSARY, featureInfo } from "../lib.js";
import Tip from "./Tip.jsx";

const FLOW = [
  { title: "8 features", sub: "physiological inputs from rPPG", icon: CircleDot },
  { title: "QAOA selection", sub: "quantum optimizer picks the strongest", icon: GitBranch },
  { title: "6 selected", sub: "compact feature subset", icon: Check },
  { title: "Quantum encoding", sub: "features mapped to qubit rotation angles", icon: Sigma },
  { title: "Hybrid VQC", sub: "PennyLane circuit + trained weights", icon: Atom },
  { title: "P(live)", sub: "output — primary classifier score", icon: Sparkles },
];

export default function QuantumFlow({ result }) {
  const quantum = result?.stages?.quantum ?? null;
  if (!quantum) return null;

  const probReal = quantum?.prob_real ?? null;
  const selected = Array.isArray(quantum?.selected_features) ? quantum.selected_features : null;

  return (
    <motion.section
      className="card qpanel"
      initial={{ opacity: 0, y: 18 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: "easeOut", delay: 0.22 }}
      aria-label="quantum classification flow — the primary classifier"
    >
      <div className="section-head">
        <span className="section-ico" aria-hidden="true">
          <Atom size={15} />
        </span>
        <div>
          <h3 className="section-title">Quantum classification</h3>
          <p className="section-sub">Hybrid quantum-classical decision flow</p>
        </div>
        <div className="section-head-right">
          <span className="q-stage">Primary classifier</span>
          <span className="chip mono">PennyLane · hybrid VQC</span>
        </div>
      </div>

      <div className="quant-flow" role="list">
        {FLOW.map((step, i) => {
          const out = i === FLOW.length - 1;
          const hl = i === 0 || i === FLOW.length - 1;
          return (
            <div className="qstep" role="listitem" key={step.title} style={{ position: "relative" }}>
              <div className={`qnode${hl ? " hl" : ""}${out ? " out" : ""}`}>
                <span className="qfrom" aria-hidden="true">
                  <step.icon size={14} />
                </span>
                <p className="qstep-title">
                  {out && probReal != null ? (
                    <span className="qout mono">
                      {probReal.toFixed(3)}
                    </span>
                  ) : (
                    step.title
                  )}
                </p>
                <p className="qstep-sub">{i === FLOW.length - 1 ? step.title : step.sub}</p>
              </div>
              {i < FLOW.length - 1 && <span className="qconn" aria-hidden="true" />}
            </div>
          );
        })}
      </div>

      {selected && (
        <div className="q-selected">
          <span className="q-selected-label">
            <Tip text={GLOSSARY.qaoa}>Selected by QAOA</Tip>
          </span>
          <div className="q-tags">
            {selected.map((name) => (
              <span className="q-tag mono" key={name}>
                <Check size={10} aria-hidden="true" />
                {featureInfo(name)?.label ?? name}
              </span>
            ))}
          </div>
        </div>
      )}

      <p className="q-note">
        <b>Primary decision layer.</b> Six of the eight rPPG features — chosen by an{" "}
        <Tip text={GLOSSARY.qaoa}>QAOA</Tip> optimizer — are encoded into a{" "}
        <Tip text={GLOSSARY.vqc}>variational quantum circuit</Tip> whose trainable weights are
        optimized classically. The circuit output is the{" "}
        <Tip text={GLOSSARY.probability}>probability of live</Tip>, which produced this verdict.
        This is the sole authority for the final decision.
      </p>
    </motion.section>
  );
}