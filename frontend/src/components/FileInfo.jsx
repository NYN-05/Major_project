import { motion } from "motion/react";
import { FileVideo, Gauge, Layers, Scan, Timer } from "lucide-react";
import { fmtClock, fmtSize } from "../lib.js";

const resolve = (v) =>
  Array.isArray(v) && v.length > 0 ? `${v[0]} × ${v[1]}` : null;

export default function FileInfo({ result }) {
  const video = result?.video;
  const frames = result?.stages?.frames ?? null;
  const rppg = result?.stages?.rppg ?? null;

  const duration = frames?.duration_s ?? frames?.stats?.duration_s;
  const size = video?.size ?? result?.video_size;
  const resolution = resolve(video?.resolution) ?? resolve(result?.resolution);
  const fps = frames?.fps ?? frames?.stats?.fps ?? rppg?.fps_used;

  const rows = [
    {
      icon: FileVideo,
      k: "File",
      v: typeof video === "string" ? video : video?.name || "—",
    },
    { icon: Timer, k: "Duration", v: duration != null ? fmtClock(duration) : "—" },
    { icon: Gauge, k: "Resolution", v: resolution || "—" },
    { icon: Layers, k: "Size", v: fmtSize(size) },
    { icon: Scan, k: "Sample rate", v: fps != null ? `${Math.round(fps)} fps` : "—" },
  ];

  return (
    <motion.section
      className="fileinfo card-soft"
      initial={{ opacity: 0, y: 18 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: "easeOut", delay: 0.24 }}
    >
      <div className="section-head">
        <span className="section-ico" aria-hidden="true">
          <FileVideo size={15} />
        </span>
        <h3 className="section-title">Analyzed file</h3>
        <span className="section-rule" aria-hidden="true" />
      </div>
      <dl className="fileinfo-grid">
        {rows.map(({ icon: Icon, k, v }) => (
          <div className="fi" key={k}>
            <dt>
              <Icon size={13} aria-hidden="true" />
              {k}
            </dt>
            <dd>{v}</dd>
          </div>
        ))}
      </dl>
    </motion.section>
  );
}