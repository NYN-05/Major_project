import { useEffect, useState } from "react";
import { motion } from "motion/react";
import { Images } from "lucide-react";
import { useThumbs } from "../hooks.js";
import { fileUrl } from "../api.js";

const BASE = (r) => {
  const s = r.replace(/\.(mp4|avi|mov)$/i, "");
  return s.replace(/[^a-z0-9-]+/gi, "_").toLowerCase();
};

const frameNum = (name) => {
  const m = String(name || "").match(/(\d+)/);
  return m ? `#${m[1]}` : null;
};

export default function FrameGallery({ result, artifacts }) {
  const [fallback, setFallback] = useState([]);
  const stem = result?.video ? BASE(String(result.video)) : null;
  const thumbs = useThumbs(artifacts, stem);

  useEffect(() => {
    if (!stem) return undefined;
    let dead = false;
    (async () => {
      try {
        const list = await artifacts(`frames/frame_sequences/${stem}/frames`);
        const files = (Array.isArray(list) ? list : list?.files ?? [])
          .filter((f) => /\.(jpe?g|png)$/i.test(f.name ?? f))
          .slice(0, 8);
        if (!dead && files.length) setFallback(files);
      } catch {
        /* no sequence dir for this run */
      }
    })();
    return () => {
      dead = true;
    };
  }, [artifacts, stem]);

  const imgs = thumbs.length ? thumbs : fallback;
  if (!imgs.length) return null;

  return (
    <motion.section
      className="card card-pad"
      initial={{ opacity: 0, y: 18 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: "easeOut", delay: 0.28 }}
    >
      <div className="section-head">
        <span className="section-ico" aria-hidden="true">
          <Images size={15} />
        </span>
        <div>
          <h3 className="section-title">Accepted frames — evidence</h3>
          <p className="section-sub">Frames that passed the stage-1 quality gate</p>
        </div>
        <div className="section-head-right">
          <span className="chip mono">{imgs.length} shown</span>
        </div>
      </div>
      <div className="gallery-grid">
        {imgs.map((f, i) => (
          <motion.figure
            className="frame-item"
            key={f.rel ?? i}
            initial={{ opacity: 0, scale: 0.97 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.03 * i, duration: 0.35, ease: "easeOut" }}
            style={{ margin: 0 }}
          >
            <img
              src={fileUrl(f.rel ?? f.name)}
              alt={`accepted frame ${i + 1}`}
              loading="lazy"
            />
            <span className="frame-idx mono">{frameNum(f.name) ?? `${i + 1}`}</span>
          </motion.figure>
        ))}
      </div>
      <p className="gallery-note">
        Frame thumbnails from the stage-1 extraction ({stem || "sequence"}). Per-frame quality
        metadata is not exposed by the API for individual thumbnails.
      </p>
    </motion.section>
  );
}