import { useEffect, useState } from "react";
import { motion } from "motion/react";
import { FileVideo, Gauge, Play, RefreshCw, Timer } from "lucide-react";
import { fmtClock, fmtSize } from "../lib.js";

export default function VideoSelected({ file, meta, onStart, onChange }) {
  const [previewUrl, setPreviewUrl] = useState(null);

  useEffect(() => {
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const duration = meta?.duration ?? null;
  const resolution = meta?.width && meta?.height ? `${meta.width} × ${meta.height}` : null;

  const rows = [
    { icon: FileVideo, k: "File", v: file.name },
    { icon: Timer, k: "Duration", v: duration != null ? fmtClock(duration) : "reading…" },
    { icon: Gauge, k: "Resolution", v: resolution || "reading…" },
    { icon: FileVideo, k: "Size", v: fmtSize(meta?.size ?? file.size) },
  ];

  return (
    <motion.section
      className="selected"
      initial={{ opacity: 0, y: 18 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.55, ease: [0.22, 1, 0.36, 1] }}
    >
      <div className="selected-preview">
        {previewUrl ? (
          <video src={previewUrl} controls muted playsInline aria-label="selected video preview" />
        ) : (
          <div className="selected-preview-ph">
            <FileVideo size={26} aria-hidden="true" />
          </div>
        )}
        <span className="selected-badge">
          <Play size={11} aria-hidden="true" />
          Ready to analyze
        </span>
      </div>

      <div className="selected-body">
        <span className="eyebrow">Video selected</span>
        <h2 className="selected-title">{file.name}</h2>
        <p className="selected-sub">
          Everything runs locally on this machine — the file is never uploaded anywhere.
        </p>

        <dl className="selected-meta">
          {rows.map(({ icon: Icon, k, v }) => (
            <div className="sm" key={k}>
              <dt>
                <Icon size={13} aria-hidden="true" />
                {k}
              </dt>
              <dd className="mono">{v}</dd>
            </div>
          ))}
        </dl>

        <div className="actions">
          <motion.button
            type="button"
            className="btn btn-primary"
            onClick={onStart}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.97 }}
          >
            <Play size={14} aria-hidden="true" />
            Start analysis
          </motion.button>
          <motion.button
            type="button"
            className="btn btn-ghost"
            onClick={onChange}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.97 }}
          >
            <RefreshCw size={14} aria-hidden="true" />
            Choose another
          </motion.button>
        </div>
      </div>
    </motion.section>
  );
}