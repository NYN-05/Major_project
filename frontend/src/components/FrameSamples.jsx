import { useEffect, useState } from "react";
import { motion } from "motion/react";
import { Images } from "lucide-react";
import { useThumbs } from "../hooks.js";

const BASE = (r) => {
  const s = r.replace(/\.(mp4|avi|mov)$/i, "");
  return s.replace(/[^a-z0-9-]+/gi, "_").toLowerCase();
};

export default function FrameSamples({ result, artifacts }) {
  const [fallback, setFallback] = useState([]);
  const stem = result?.video ? BASE(String(result.video)) : null;
  const thumbs = useThumbs(artifacts, stem);

  useEffect(() => {
    if (!stem) return undefined;
    let dead = false;
    (async () => {
      try {
        const list = await artifacts(`frames/frame_sequences/${stem}/frames`);
        const imgs = (Array.isArray(list) ? list : list?.files ?? [])
          .filter((f) => /\.(jpe?g|png)$/i.test(f))
          .slice(0, 5);
        if (!dead && imgs.length) setFallback(imgs);
      } catch {
        /* restore path: no sequence dir for this run */
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
      className="samples card-soft"
      initial={{ opacity: 0, y: 18 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: "easeOut", delay: 0.28 }}
    >
      <div className="section-head">
        <span className="section-ico" aria-hidden="true">
          <Images size={15} />
        </span>
        <h3 className="section-title">Sample frames</h3>
        <span className="section-rule" aria-hidden="true" />
        <span className="chip mono">{imgs.length} kept</span>
      </div>
      <div className="samples-grid">
        {imgs.map((img, i) => (
          <motion.figure
            className="sample-frame"
            key={i}
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.05 * i, duration: 0.4, ease: "easeOut" }}
          >
            <img src={img} alt={`accepted frame ${i + 1}`} loading="lazy" />
          </motion.figure>
        ))}
      </div>
    </motion.section>
  );
}