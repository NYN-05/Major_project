import { motion } from "motion/react";
import { FileVideo, Gauge, Layers, MonitorPlay, Scan } from "lucide-react";
import { fmtClock, fmtSize } from "../lib.js";

export default function FileInfo({ result }) {
  const video = result?.video;
  const vm = result?.video_meta ?? {};
  const frames = result?.stages?.frames ?? null;
  const rppg = result?.stages?.rppg ?? null;

  const name = vm?.name ?? (typeof video === "string" ? video : video?.name) ?? "—";
  const duration = vm?.duration_s ?? frames?.duration_s ?? frames?.stats?.duration_s;
  const size = vm?.size_bytes ?? video?.size ?? result?.video_size;
  const resolution =
    vm?.width && vm?.height ? `${vm.width} × ${vm.height}` : null;
  const fps = vm?.fps ?? frames?.fps ?? frames?.stats?.fps ?? rppg?.fps_used;
  const frameCount = vm?.frame_count ?? null;

  const rows = [
    { icon: FileVideo, k: "File", v: name },
    { icon: MonitorPlay, k: "Duration", v: duration != null ? fmtClock(duration) : "—" },
    { icon: Gauge, k: "Resolution", v: resolution || "—" },
    { icon: Layers, k: "Size", v: fmtSize(size) },
    {
      icon: Scan,
      k: "Sample rate",
      v: fps != null ? `${Math.round(fps)} fps` : "—",
    },
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
        {frameCount != null && <span className="chip mono">{frameCount} source frames</span>}
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