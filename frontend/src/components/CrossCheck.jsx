import { motion } from "motion/react";
import { BrainCircuit } from "lucide-react";
import { GLOSSARY } from "../lib.js";
import Tip from "./Tip.jsx";

export default function CrossCheck({ result }) {
  const xc = result?.stages?.rppg_crosscheck ?? null;
  if (!xc?.verdict) return null;

  const fake = xc.verdict === "DEEPFAKE";
  const tone = fake ? "bad" : "ok";

  return (
    <motion.section
      className="card crosscheck"
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: "easeOut", delay: 0.26 }}
      aria-label="secondary ML cross-check result"
    >
      <span className="cc-ico" aria-hidden="true">
        <BrainCircuit size={16} />
      </span>
      <div className="cc-body">
        <p className="cc-title">Secondary ML cross-check</p>
        <p className="cc-sub">
          <Tip text={GLOSSARY.crosscheck}>
            Random-Forest model on the same rPPG features — informational, not the final authority.
          </Tip>
        </p>
      </div>
      <div className="cc-value">
        <span className={`cc-verdict ${tone}`}>
          {fake ? "Deepfake signal" : "Real-person signal"}
        </span>
        <span className="cc-prob mono">
          probability <b>{xc.probability != null ? xc.probability.toFixed(3) : "—"}</b>{" "}
          {fake ? "of being fake" : "of being real"}
        </span>
      </div>
    </motion.section>
  );
}