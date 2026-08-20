import { useEffect, useRef } from "react";
import { motion, useReducedMotion } from "motion/react";
import { HeartPulse, Waves } from "lucide-react";
import { ROIS, fmtVal, GLOSSARY } from "../lib.js";
import Tip from "./Tip.jsx";

function Waveform({ data, tone, reduced }) {
  const ref = useRef(null);

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

    const finite = data.filter((v) => typeof v === "number" && Number.isFinite(v));
    if (!finite.length) return;
    const lo = Math.min(...finite);
    const hi = Math.max(...finite);
    const mid = (lo + hi) / 2;
    const half = Math.max((hi - lo) / 2, 1e-9);
    const pad = 14;
    const ys = data.map((v) =>
      typeof v === "number" && Number.isFinite(v) ? h / 2 + ((v - mid) / half) * (h / 2 - pad) : null
    );
    const xs = (i) => (ys.length > 1 ? (i / (ys.length - 1)) * w : w / 2);

    const color =
      tone === "ok"
        ? { line: "#a3e635", fill: "rgba(163, 230, 53, 0.10)" }
        : tone === "bad"
          ? { line: "#ef4444", fill: "rgba(239, 68, 68, 0.09)" }
          : { line: "#f59e0b", fill: "rgba(245, 158, 11, 0.09)" };

    const drawTo = (frac) => {
      const count = Math.max(1, Math.floor((ys.length - 1) * frac));
      ctx.clearRect(0, 0, w, h);
      const grad = ctx.createLinearGradient(0, 0, w, 0);
      grad.addColorStop(0, color.line);
      grad.addColorStop(1, color.line);

      ctx.beginPath();
      ctx.moveTo(0, h);
      for (let i = 0; i <= count; i += 1) {
        const y = ys[i];
        if (y === null) continue;
        ctx.lineTo(xs(i), y);
      }
      ctx.lineTo(xs(count), h);
      ctx.closePath();
      ctx.fillStyle = color.fill;
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
      ctx.lineWidth = 1.8;
      ctx.lineJoin = "round";
      ctx.stroke();
    };

    if (reduced) {
      drawTo(1);
      return undefined;
    }
    let raf;
    const t0 = performance.now();
    const dur = 1100;
    const step = (t) => {
      const e = 1 - Math.pow(1 - Math.min(1, (t - t0) / dur), 3);
      drawTo(e);
      if (e < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [data, tone, reduced]);

  return <canvas ref={ref} className="signal-canvas" />;
}

export default function SignalPanel({ signalData, result }) {
  const reduced = useReducedMotion();
  const data = signalData?.signal;
  const rppg = result?.stages?.rppg ?? null;
  const features = rppg?.features ?? null;
  const verdictTone = (result?.verdict?.label ?? "UNCERTAIN") === "REAL" ? "ok" : result?.verdict?.label === "FAKE" ? "bad" : "warn";

  const bpm = features?.heart_rate_bpm ?? null;
  const sqi = features?.signal_quality_index ?? null;
  const error = signalData?.error ?? null;

  return (
    <motion.section
      className="card signal-panel"
      initial={{ opacity: 0, y: 18 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: "easeOut", delay: 0.18 }}
      aria-label="rPPG physiological signal analysis"
    >
      <div className="section-head">
        <span className="section-ico" aria-hidden="true">
          <Waves size={15} />
        </span>
        <div>
          <h3 className="section-title">rPPG analysis</h3>
          <p className="section-sub">
            Reconstructed physiological signal — blended from {ROIS.length} facial ROIs
          </p>
        </div>
        <div className="section-head-right">
          {bpm != null && (
            <span className="chip chip-lime">
              <HeartPulse size={11} aria-hidden="true" />
              ≈ {Math.round(bpm)} BPM
            </span>
          )}
          {sqi != null && (
            <Tip text={GLOSSARY.sqi}>
              <span className="chip">SQI {fmtVal(sqi, 2)}</span>
            </Tip>
          )}
        </div>
      </div>

      <div className="signal-stage">
        {data && data.length > 0 ? (
          <Waveform data={data} tone={verdictTone} reduced={reduced} />
        ) : (
          <p
            className="mono"
            style={{
              position: "absolute",
              inset: 0,
              display: "grid",
              placeItems: "center",
              margin: 0,
              fontSize: 11.5,
              color: "var(--dim)",
              textAlign: "center",
              padding: "0 20px",
            }}
          >
            {error ? `waveform unavailable — ${error}` : "waveform is being reconstructed…"}
          </p>
        )}
      </div>

      <div className="signal-legend">
        <p className="signal-cap mono">
          pulse trace reconstructed from the accepted frames · color scales with the verdict
        </p>
        <div className="roi-chips">
          <span className="roi-chip" style={{ color: "var(--dim)" }}>ROI blend</span>
          {ROIS.map((r) => (
            <span className={`roi-chip ${r.tone}`} key={r.name}>
              <i aria-hidden="true" />
              {r.name} {r.weight}%
            </span>
          ))}
        </div>
      </div>
    </motion.section>
  );
}