import { useCallback, useEffect, useRef, useState } from "react";
import { artifacts, detect, fileUrl, health, previous, stream } from "./api.js";

/* ------------------------------------------------------------------ */
/* constant meta                                                       */
/* ------------------------------------------------------------------ */

const STAGES = ["01 FRAMES", "02 PHYSIOLOGY", "03 QUANTUM"];

const FEATURE_META = [
  ["heart_rate_bpm", "Heart rate", "BPM"],
  ["snr_db", "Signal-to-noise", "dB"],
  ["prv_std_ms", "Pulse-rate variability", "ms"],
  ["spectral_entropy", "Spectral entropy", "nats"],
  ["mad", "Mean absolute deviation", "a.u."],
  ["signal_quality_index", "Signal quality index", "0–1"],
  ["cheek_forehead_correlation", "Cheek ↔ forehead correlation", "r"],
  ["left_right_cheek_correlation", "Left ↔ right cheek correlation", "r"],
];

const DECISION = {
  REAL: { word: "Verified", tone: "ok", note: "Physiological signal consistent with a live subject." },
  FAKE: { word: "Rejected", tone: "bad", note: "Physiological evidence inconsistent with a live subject." },
  UNCERTAIN: { word: "Needs review", tone: "warn", note: "Below the confidence floor. Escalate for manual review." },
};

/* ------------------------------------------------------------------ */
/* hooks                                                               */
/* ------------------------------------------------------------------ */

function usePrefersReducedMotion() {
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    setReduced(mq.matches);
    const cb = (e) => setReduced(e.matches);
    mq.addEventListener("change", cb);
    return () => mq.removeEventListener("change", cb);
  }, []);
  return reduced;
}

function useElapsed(active) {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    if (!active) return undefined;
    const t0 = Date.now();
    const id = setInterval(() => setElapsed((Date.now() - t0) / 1000), 1000);
    return () => clearInterval(id);
  }, [active]);
  return elapsed;
}

const fmtClock = (s) => {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
};

const fmtVal = (v, digits = 3) => {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  if (typeof v !== "number") return String(v);
  return v.toFixed(digits);
};

/* ------------------------------------------------------------------ */
/* waveform                                                            */
/* ------------------------------------------------------------------ */

function Waveform({ signal, caption }) {
  const ref = useRef(null);
  const reduced = usePrefersReducedMotion();

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    ctx.strokeStyle = "rgba(74, 102, 136, 0.22)";
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

    const data = Array.isArray(signal) ? signal : [];
    if (data.length === 0) {
      ctx.strokeStyle = "rgba(132, 150, 172, 0.5)";
      ctx.setLineDash([4, 6]);
      ctx.beginPath();
      ctx.moveTo(0, h / 2);
      ctx.lineTo(w, h / 2);
      ctx.stroke();
      ctx.setLineDash([]);
      return;
    }

    const finite = data.filter((v) => typeof v === "number" && Number.isFinite(v));
    const lo = Math.min(...finite);
    const hi = Math.max(...finite);
    const mid = (lo + hi) / 2;
    const half = Math.max((hi - lo) / 2, 1e-9);
    const pad = 10;
    const ys = data.map((v) =>
      typeof v === "number" && Number.isFinite(v)
        ? h / 2 + ((v - mid) / half) * (h / 2 - pad)
        : null
    );

    const drawTo = (frac) => {
      const count = Math.max(1, Math.floor((ys.length - 1) * frac));
      ctx.strokeStyle = "#F05050";
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      let started = false;
      for (let i = 0; i <= count; i += 1) {
        const y = ys[i];
        if (y === null) {
          started = false;
          continue;
        }
        const x = ys.length > 1 ? (i / (ys.length - 1)) * w : w / 2;
        if (!started) {
          ctx.moveTo(x, y);
          started = true;
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
    };

    if (reduced) {
      drawTo(1);
      return;
    }
    let raf;
    const t0 = performance.now();
    const dur = 1200;
    const step = (t) => {
      const p = Math.min(1, (t - t0) / dur);
      const ease = 1 - Math.pow(1 - p, 3);
      ctx.clearRect(0, 0, w, h);
      for (let x = 0; x <= w; x += 26) {
        ctx.strokeStyle = "rgba(74, 102, 136, 0.22)";
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
        ctx.stroke();
      }
      for (let y = 0; y <= h; y += 26) {
        ctx.strokeStyle = "rgba(74, 102, 136, 0.22)";
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
      }
      drawTo(ease);
      if (p < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [signal, reduced]);

  return (
    <figure className="wave">
      <canvas ref={ref} className="wave-canvas" />
      <figcaption className="wave-cap">{caption}</figcaption>
    </figure>
  );
}

/* ------------------------------------------------------------------ */
/* verdict rig                                                         */
/* ------------------------------------------------------------------ */

function VerdictRig({ result, signalData }) {
  const label = result?.verdict?.label ?? "UNCERTAIN";
  const dec = DECISION[label] ?? DECISION.UNCERTAIN;
  const confidence = result?.verdict?.confidence;
  const probReal = result?.stages?.quantum?.prob_real;
  const reason = result?.verdict?.reason;

  return (
    <section className="rig" aria-label="verdict">
      <div className="rig-verdict">
        <span className="eyebrow mono">FINAL DECISION</span>
        <h2 className={`verdict-word tone-${dec.tone}`}>{dec.word}</h2>
        {confidence !== undefined && confidence !== null && (
          <p className="verdict-conf mono">
            confidence {fmtVal(confidence, 4)} · P(real) {fmtVal(probReal, 4)}
          </p>
        )}
        <p className="verdict-note">{reason || dec.note}</p>
      </div>

      <div className="rig-signal">
        <Waveform
          signal={signalData?.signal}
          caption={
            signalData
              ? `∿ combined rPPG pulse · POS · ${signalData.fps ?? "—"} fps · ${signalData.n ?? "—"} usable frames`
              : "∿ waveform unavailable — pulse trace not archived for this run"
          }
        />
        {probReal !== undefined && (
          <div className="gauge">
            <div className="gauge-scale">
              <div className="gauge-fill" style={{ width: `${Math.max(0, Math.min(1, probReal)) * 100}%` }} />
              <span className="tick tick-l" style={{ left: "30%" }} />
              <span className="tick tick-r" style={{ left: "70%" }} />
            </div>
            <div className="gauge-labels mono">
              <span>FAKE</span>
              <span style={{ left: "30%" }}>0.30</span>
              <span style={{ left: "70%" }}>0.70</span>
              <span>REAL</span>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ */
/* evidence cards                                                      */
/* ------------------------------------------------------------------ */

function Card({ no, title, children, wide }) {
  return (
    <section className={`card${wide ? " card-wide" : ""}`}>
      <header className="card-head">
        <span className="card-no mono">{no}</span>
        <h3 className="card-title">{title}</h3>
      </header>
      <div className="card-body">{children}</div>
    </section>
  );
}

function Stat({ k, v }) {
  return (
    <div className="stat">
      <span className="stat-k mono">{k}</span>
      <span className="stat-v">{v}</span>
    </div>
  );
}

function FramesCard({ frames, videoStem }) {
  const [thumbs, setThumbs] = useState([]);

  useEffect(() => {
    let alive = true;
    if (!videoStem) {
      setThumbs([]);
      return undefined;
    }
    artifacts(`frames/frame_sequences/${videoStem}/frames`).then(({ files }) => {
      if (alive) setThumbs(files.slice(0, 12));
    });
    return () => {
      alive = false;
    };
  }, [videoStem]);

  if (!frames || frames.status !== "success") {
    return (
      <Card no="01" title="Frames">
        <p className="empty-note">No frame-stage evidence for this result.</p>
      </Card>
    );
  }
  const s = frames.stats ?? {};
  const rejections = s.rejections ?? {};
  const reasons = Object.entries(rejections);

  return (
    <Card no="01" title="Frames">
      <div className="stat-grid">
        <Stat k="accepted" v={`${s.accepted_frames ?? 0} / ${s.sampled_frames ?? 0}`} />
        <Stat k="mean quality" v={fmtVal(s.mean_quality_score, 3)} />
        <Stat k="temporal coverage" v={`${fmtVal((s.temporal_coverage_ratio ?? 0) * 100, 0)}%`} />
      </div>
      {reasons.length > 0 && (
        <div className="chips">
          {reasons.map(([k, v]) => (
            <span key={k} className="chip chip-warn mono">
              {k} ×{v}
            </span>
          ))}
        </div>
      )}
      {thumbs.length > 0 ? (
        <div className="thumbs">
          {thumbs.map((f) => (
            <img key={f.rel} src={fileUrl(f.rel)} alt={f.name} loading="lazy" />
          ))}
        </div>
      ) : (
        <p className="empty-note">Accepted frames not on disk for this run.</p>
      )}
    </Card>
  );
}

function PhysiologyCard({ rppg, hasPlot, plotRel }) {
  if (!rppg || !rppg.features) {
    return (
      <Card no="02" title="Physiology">
        <p className="empty-note">No physiological evidence — insufficient usable frames.</p>
      </Card>
    );
  }
  const feats = rppg.features;
  return (
    <Card no="02" title="Physiology">
      <div className="stat-grid">
        <Stat k="input mode" v={rppg.input_mode === "stage1_frames" ? "stage-1 frames" : "direct video"} />
        <Stat k="usable frames" v={`${rppg.n_frames_usable}/${rppg.n_frames_total}`} />
        <Stat k="sampling" v={`${fmtVal(rppg.fps_used, 1)} fps`} />
      </div>
      <dl className="feat">
        {FEATURE_META.map(([key, name, unit]) => (
          <div key={key} className="feat-row">
            <dt>{name}</dt>
            <dd className="mono">
              {fmtVal(feats[key])}
              <span className="feat-unit">{unit}</span>
            </dd>
            <span className="feat-key mono">{key}</span>
          </div>
        ))}
      </dl>
      {hasPlot && (
        <img className="plot" src={fileUrl(plotRel)} alt="rPPG diagnostic plot" loading="lazy" />
      )}
    </Card>
  );
}

function QuantumCard({ quantum, plots, rppgCrosscheck }) {
  if (!quantum) {
    return (
      <Card no="03" title="Quantum">
        <p className="empty-note">No quantum-stage evidence for this result.</p>
      </Card>
    );
  }
  return (
    <Card no="03" title="Quantum">
      <div className="stat-grid">
        <Stat k="P(real)" v={fmtVal(quantum.prob_real, 4)} />
        <Stat k="verdict" v={quantum.verdict} />
        <Stat k="confidence" v={fmtVal(quantum.confidence, 4)} />
      </div>
      {Array.isArray(quantum.selected_features) && quantum.selected_features.length > 0 && (
        <div className="chips">
          {quantum.selected_features.map((f) => (
            <span key={f} className="chip chip-quantum mono">
              {f}
            </span>
          ))}
        </div>
      )}
      {rppgCrosscheck && rppgCrosscheck.verdict && (
        <p className="cross mono">
          cross-check · random forest →{" "}
          <b className={rppgCrosscheck.verdict === "DEEPFAKE" ? "tone-bad" : ""}>
            {rppgCrosscheck.verdict}
          </b>{" "}
          ({fmtVal(rppgCrosscheck.probability, 4)})
        </p>
      )}
      {plots.length > 0 && (
        <div className="plots">
          {plots.map((p) => (
            <img key={p.rel} className="plot" src={fileUrl(p.rel)} alt={p.name} loading="lazy" />
          ))}
        </div>
      )}
    </Card>
  );
}

/* ------------------------------------------------------------------ */
/* upload bay + progress                                               */
/* ------------------------------------------------------------------ */

function UploadBay({ phase, lines, stageIdx, elapsed, onFile, onCancel }) {
  const [drag, setDrag] = useState(false);
  const inputRef = useRef(null);
  const consoleRef = useRef(null);

  useEffect(() => {
    const el = consoleRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [lines]);

  const pick = (files) => {
    const f = files?.[0];
    if (f && f.type.startsWith("video/")) onFile(f);
  };

  if (phase === "running") {
    return (
      <section className="bay bay-running">
        <div className="run-head">
          <span className="live-dot" aria-hidden="true" />
          <span className="mono run-title">ANALYSIS IN PROGRESS — {fmtClock(elapsed)}</span>
        </div>
        <div className="rail" role="status" aria-live="polite">
          {STAGES.map((name, i) => (
            <span
              key={name}
              className={`rail-step mono${i < stageIdx ? " done" : ""}${i === stageIdx ? " active" : ""}`}
            >
              <span className="rail-dot" />
              {name}
            </span>
          ))}
        </div>
        <div className="console mono" ref={consoleRef} aria-label="pipeline log">
          {lines.slice(-10).map((l, i) => (
            <div key={i} className={`console-line${l.includes("[") ? " bright" : ""}`}>
              {l}
            </div>
          ))}
          <div className="console-line bright caret">▌</div>
        </div>
      </section>
    );
  }

  return (
    <section
      className={`bay${drag ? " bay-drag" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        setDrag(true);
      }}
      onDragLeave={() => setDrag(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDrag(false);
        pick(e.dataTransfer.files);
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept="video/*"
        hidden
        onChange={(e) => pick(e.target.files)}
      />
      <span className="bay-glyph" aria-hidden="true">
        ∿
      </span>
      <h2 className="bay-title">{phase === "error" ? "Analysis failed" : "Drop a KYC video to begin"}</h2>
      <p className="bay-sub">
        {phase === "error"
          ? "The last run errored on the server. Check the backend console, then try again."
          : "mp4 · avi · mov — sampled at 10 fps, scored across the three stages, processed entirely on this machine."}
      </p>
      <button className="btn btn-primary" onClick={() => inputRef.current?.click()}>
        Choose video
      </button>
      <p className="bay-hint mono">or drag the file onto this panel</p>
    </section>
  );
}

/* ------------------------------------------------------------------ */
/* app                                                                 */
/* ------------------------------------------------------------------ */

export default function App() {
  const [phase, setPhase] = useState("idle"); // idle | running | done | error
  const [result, setResult] = useState(null);
  const [signalData, setSignalData] = useState(null);
  const [lines, setLines] = useState([]);
  const [stageIdx, setStageIdx] = useState(0);
  const [videoName, setVideoName] = useState(null);
  const [runError, setRunError] = useState(null);
  const [healthInfo, setHealth] = useState(null);
  const elapsed = useElapsed(phase === "running");

  useEffect(() => {
    health().then(setHealth);
    previous().then(({ result: prev }) => {
      if (prev) {
        setResult(prev);
        setVideoName(prev.video);
      }
    });
  }, []);

  useEffect(() => {
    let alive = true;
    const sig = result?._signal;
    if (!sig) {
      setSignalData(null);
      return undefined;
    }
    fetch(`/api/files?rel=${encodeURIComponent(sig)}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (alive) setSignalData(d);
      })
      .catch(() => {
        if (alive) setSignalData(null);
      });
    return () => {
      alive = false;
    };
  }, [result]);

  const run = useCallback((file) => {
    setPhase("running");
    setResult(null);
    setSignalData(null);
    setLines([]);
    setStageIdx(0);
    setVideoName(file.name);
    setRunError(null);
    detect(file)
      .then(({ job }) => {
        stream(job, {
          line: (l) => setLines((prev) => [...prev.slice(-250), l]),
          stage: (l) => {
            const m = l.match(/\[(\d)\/3\]/);
            if (m) setStageIdx(parseInt(m[1], 10));
          },
          result: (res) => {
            setResult(res);
            setStageIdx(3);
            setPhase("done");
            health().then(setHealth);
          },
          error: (msg) => {
            setRunError(msg);
            setPhase("error");
          },
        });
      })
      .catch((err) => {
        setRunError(String(err));
        setPhase("error");
      });
  }, []);

  const highlights = healthInfo?.artifacts?.quantum_plots ?? [];
  const quantumPlots = highlights.map((n) => ({ name: n, rel: `quantum/${n}` }));
  const rppgPlotRel = healthInfo?.artifacts?.rppg_png ? "rppg/rppg_output.png" : null;
  const videoStem = result?.video ? result.video.replace(/\.\w+$/, "") : null;

  const haveResult = phase === "done" && result;
  const showEvidence = haveResult || phase === "idle";

  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-glyph" aria-hidden="true">
            ∿
          </span>
          <div>
            <h1>
              rPPG·QC <span className="brand-amp">∇</span> QUANTUM
            </h1>
            <p className="mono">DEEPFAKE VERIFIER — KYC</p>
          </div>
        </div>
        <div className="topbar-right mono">
          <span className="ok-dot" aria-hidden="true" />
          LOCAL PROCESSING · VIDEO NEVER LEAVES THIS MACHINE
        </div>
      </header>

      <main>
        {phase === "error" && <div className="error-banner mono">{runError}</div>}

        <UploadBay
          phase={phase}
          lines={lines}
          stageIdx={stageIdx}
          elapsed={elapsed}
          onFile={run}
        />

        {haveResult && <VerdictRig result={result} signalData={signalData} />}

        {showEvidence && (
          <div className="cards">
            <FramesCard frames={result?.stages?.frames} videoStem={videoStem} />
            <PhysiologyCard
              rppg={result?.stages?.rppg}
              hasPlot={Boolean(rppgPlotRel)}
              plotRel={rppgPlotRel || ""}
            />
            <QuantumCard
              quantum={result?.stages?.quantum}
              plots={quantumPlots}
              rppgCrosscheck={result?.stages?.rppg_crosscheck}
            />
          </div>
        )}

        {!showEvidence && (
          <section className="card card-wide">
            <p className="empty-note">
              No evidence on file. Run an analysis to generate the dossier below.
            </p>
          </section>
        )}
      </main>

      <footer className="foot mono">
        <span>FRAMES → rPPG PULSE → QAOA SUBSET → HYBRID VQC → P(REAL)</span>
        <span>REAL ≥ 0.70 · FAKE ≤ 0.30 · ELSE UNCERTAIN</span>
      </footer>
    </div>
  );
}