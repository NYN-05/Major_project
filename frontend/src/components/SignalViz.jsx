import { useEffect, useRef } from "react";
import { motion, useReducedMotion } from "motion/react";
import { HeartPulse, Waves } from "lucide-react";

export default function SignalViz({ signalData, bpm, sqi }) {
  const ref = useRef(null);
  const reduced = useReducedMotion();
  const data = signalData?.signal;

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || !Array.isArray(data) || data.length === 0) return;
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    const grid = (color) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      for (let x = 0; x <= w; x += 26) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
        ctx.stroke();
      }
      for (let y = 0; y <= h; y += 26) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
      }
    };

    const finite = data.filter((v) => typeof v === "number" && Number.isFinite(v));
    const lo = Math.min(...finite);
    const hi = Math.max(...finite);
    const mid = (lo + hi) / 2;
    const half = Math.max((hi - lo) / 2, 1e-9);
    const pad = 14;
    const ys = data.map((v) =>
      typeof v === "number" && Number.isFinite(v) ? h / 2 + ((v - mid) / half) * (h / 2 - pad) : null
    );
    const xs = (i) => (ys.length > 1 ? (i / (ys.length - 1)) * w : w / 2);

    const drawTo = (frac) => {
      const count = Math.max(1, Math.floor((ys.length - 1) * frac));
      const grad = ctx.createLinearGradient(0, 0, w, 0);
      grad.addColorStop(0, "#f87171");
      grad.addColorStop(0.6, "#fb923c");
      grad.addColorStop(1, "#fbbf24");

      ctx.beginPath();
      ctx.moveTo(0, h);
      for (let i = 0; i <= count; i += 1) {
        const y = ys[i];
        if (y === null) continue;
        ctx.lineTo(xs(i), y);
      }
      ctx.lineTo(xs(count), h);
      ctx.closePath();
      ctx.fillStyle = "rgba(248, 113, 113, 0.08)";
      ctx.fill();

      ctx.beginPath();
      let started = false;
      for (let i = 0; i <= count; i += 1) {
        const y = ys[i];
        if (y === null) {
          started = false;
          continue;
        }
        if (!started) {
          ctx.moveTo(xs(i), y);
          started = true;
        } else {
          ctx.lineTo(xs(i), y);
        }
      }
      ctx.strokeStyle = grad;
      ctx.lineWidth = 2;
      ctx.lineJoin = "round";
      ctx.shadowColor = "rgba(248, 113, 113, 0.45)";
      ctx.shadowBlur = 12;
      ctx.stroke();
      ctx.shadowBlur = 0;
    };

    if (reduced) {
      grid("rgba(148, 163, 184, 0.12)");
      drawTo(1);
      return undefined;
    }
    grid("rgba(148, 163, 184, 0.12)");
    let raf;
    const t0 = performance.now();
    const dur = 1400;
    const step = (t) => {
      const p = Math.min(1, (t - t0) / dur);
      const ease = 1 - Math.pow(1 - p, 3);
      ctx.clearRect(0, 0, w, h);
      grid("rgba(148, 163, 184, 0.12)");
      drawTo(ease);
      if (p < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [data, reduced]);

  if (!data || data.length === 0) return null;

  return (
    <motion.section
      className="signal card-soft"
      initial={{ opacity: 0, y: 18 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: "easeOut", delay: 0.16 }}
    >
      <div className="section-head">
        <span className="section-ico" aria-hidden="true">
          <Waves size={15} />
        </span>
        <h3 className="section-title">Extracted rPPG signal</h3>
        <span className="section-rule" aria-hidden="true" />
        <div className="signal-chips">
          {bpm != null && (
            <span className="chip chip-signal">
              <HeartPulse size={11} aria-hidden="true" />
              ≈ {Math.round(bpm)} BPM
            </span>
          )}
          {sqi != null && (
            <span className="chip chip-signal">quality {fmt(sqi)}</span>
          )}
        </div>
      </div>
      <div className="signal-stage">
        <canvas ref={ref} className="signal-canvas" />
        <p className="signal-cap mono">
          measured pulse trace from the accepted frames — temporal physiological consistency of the face
        </p>
      </div>
    </motion.section>
  );
}

const fmt = (v) => (v == null ? "—" : Number(v).toFixed(2));