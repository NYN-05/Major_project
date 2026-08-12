import { useState } from "react";
import { motion } from "motion/react";
import { ChevronDown, FileText, FlaskConical, MonitorPlay } from "lucide-react";
import { fmtClock, fmtSize, fmtVal } from "../lib.js";

function Collapse({ title, icon: Icon, children, defaultOpen = false }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="details-item">
      <button
        type="button"
        className="card-head"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
      >
        <Icon size={14} aria-hidden="true" />
        {title}
        <ChevronDown size={14} className="chev" aria-hidden="true" />
      </button>
      <motion.div
        initial={false}
        animate={{ height: open ? "auto" : 0, opacity: open ? 1 : 0 }}
        transition={{ duration: 0.26, ease: [0.22, 1, 0.36, 1] }}
        style={{ overflow: "hidden" }}
      >
        {open && children}
      </motion.div>
    </div>
  );
}

export default function TechnicalDetails({ result, videoMeta }) {
  const rppg = result?.stages?.rppg ?? null;
  const frames = result?.stages?.frames?.stats ?? null;
  const quantum = result?.stages?.quantum ?? null;
  const video = result?.video ?? "—";
  const ts = result?.timestamp ?? null;

  const rows = [
    { k: "Source video", v: video },
    { k: "Analyzed at", v: ts ? new Date(ts).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "medium" }) : "—" },
    { k: "Duration", v: videoMeta?.duration != null ? fmtClock(videoMeta.duration) : "—" },
    { k: "Resolution", v: videoMeta?.width && videoMeta?.height ? `${videoMeta.width} × ${videoMeta.height}` : "—" },
    { k: "File size", v: fmtSize(videoMeta?.size) },
    { k: "rPPG method", v: rppg?.method ?? "—" },
    { k: "rPPG input", v: rppg?.input_mode ?? "—" },
    { k: "Sample rate", v: rppg?.fps_used != null ? `${Math.round(rppg.fps_used)} fps` : "—" },
    { k: "Usable frames", v: rppg ? `${rppg.n_frames_usable}/${rppg.n_frames_total}` : "—" },
  ];

  const frameRows = frames
    ? [
        { k: "Sampled frames", v: String(frames.sampled_frames ?? "—") },
        { k: "Accepted frames", v: String(frames.accepted_frames ?? "—") },
        { k: "Mean quality score", v: fmtVal(frames.mean_quality_score, 3) },
        { k: "Mean face confidence", v: fmtVal(frames.mean_face_confidence, 3) },
        { k: "Temporal coverage", v: fmtVal(frames.temporal_coverage_ratio, 3) },
      ]
    : [];

  const rejections = frames?.rejections
    ? Object.entries(frames.rejections).map(([k, v]) => ({ k, v: String(v) }))
    : [];

  const quantumRows = quantum
    ? [
        { k: "Confidence", v: fmtVal(quantum.confidence, 4) },
        { k: "Probability of live", v: fmtVal(quantum.prob_real, 4) },
        { k: "Quantum verdict", v: quantum.verdict ?? "—" },
        { k: "Selected indices", v: Array.isArray(quantum.selected_indices) ? quantum.selected_indices.join(", ") : "—" },
        { k: "Feature scaler", v: typeof quantum.scaler_file === "string" ? quantum.scaler_file.split(String.fromCharCode(92)).pop() : "—" },
      ]
    : [];

  return (
    <section className="card card-pad" aria-label="technical details">
      <div className="section-head">
        <span className="section-ico" aria-hidden="true">
          <MonitorPlay size={15} />
        </span>
        <div>
          <h3 className="section-title">Technical details</h3>
          <p className="section-sub">Raw pipeline facts — for the technical reviewer</p>
        </div>
      </div>

      <div className="details-stack">
        <Collapse title="File & processing metadata" icon={FileText}>
          <dl className="dl">
            {rows.map((r) => (
              <div className="dl-row" key={r.k}>
                <dt>{r.k}</dt>
                <dd className="mono">{r.v}</dd>
              </div>
            ))}
          </dl>
        </Collapse>

        {frameRows.length > 0 && (
          <Collapse title="Frame quality stage" icon={FileText}>
            <dl className="dl">
              {frameRows.map((r) => (
                <div className="dl-row" key={r.k}>
                  <dt>{r.k}</dt>
                  <dd className="mono">{r.v}</dd>
                </div>
              ))}
              {rejections.length > 0 && (
                <div className="dl-row" style={{ gridColumn: "1 / -1", flexWrap: "wrap" }}>
                  <dt>Rejection by reason</dt>
                  <dd>
                    <span className="rej-tags">
                      {rejections.map((r) => (
                        <span key={r.k} className="rej-tag mono">
                          {r.k} × {r.v}
                        </span>
                      ))}
                    </span>
                  </dd>
                </div>
              )}
            </dl>
          </Collapse>
        )}

        {quantumRows.length > 0 && (
          <Collapse title="Quantum model & artifacts" icon={FlaskConical}>
            <dl className="dl">
              {quantumRows.map((r) => (
                <div className="dl-row" key={r.k}>
                  <dt>{r.k}</dt>
                  <dd className="mono">{r.v}</dd>
                </div>
              ))}
            </dl>
          </Collapse>
        )}
      </div>
    </section>
  );
}