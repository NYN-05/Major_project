import { useCallback, useEffect, useRef, useState } from "react";
import { artifacts, detect, fileUrl, previous, stream } from "./api.js";
import ProgressBar from "./components/ProgressBar.jsx";

/* ------------------------------------------------------------------ */
/* constant meta                                                       */
/* ------------------------------------------------------------------ */

const SIMPLE_STEPS = [
  ["Checking the video quality", "Splitting the video into frames and keeping only the clear, usable ones."],
  ["Measuring the pulse from the face", "Tracking tiny skin-color changes — a live person's face shows a very small pulse."],
  ["Running the final check", "Combining every measurement into a single verdict."],
];

const PLAIN = {
  REAL: {
    word: "Looks authentic",
    icon: "✓",
    tone: "ok",
    note: "This video shows a real, live person. The natural pulse of the face matches a genuine recording.",
  },
  FAKE: {
    word: "Likely AI-generated",
    icon: "✗",
    tone: "bad",
    note: "This video does not show the natural pulse of a live person — typical of AI-generated or manipulated footage.",
  },
  UNCERTAIN: {
    word: "Can't decide — needs a human check",
    icon: "?",
    tone: "warn",
    note: "The signal was too weak for a confident decision. A manual review is recommended.",
  },
};

const confWord = (c) => (c >= 0.6 ? "High confidence" : c >= 0.3 ? "Moderate confidence" : "Low confidence");

/* ------------------------------------------------------------------ */
/* hooks                                                               */
/* ------------------------------------------------------------------ */

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

function useTheme() {
  const [theme, setTheme] = useState(() => {
    try {
      const saved = localStorage.getItem("rppgqc.theme");
      if (saved === "light" || saved === "dark") return saved;
    } catch {
      /* storage unavailable */
    }
    return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  });

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
    try {
      localStorage.setItem("rppgqc.theme", theme);
    } catch {
      /* storage unavailable */
    }
  }, [theme]);

  return [theme, setTheme];
}

function useThumbs(videoStem) {
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
  return thumbs;
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
/* simple view                                                         */
/* ------------------------------------------------------------------ */

function VideoStrip({ videoName, meta, elapsed }) {
  if (!videoName) return null;
  const parts = [];
  if (meta?.duration) parts.push(fmtClock(meta.duration));
  if (meta?.size) parts.push(`${(meta.size / 1048576).toFixed(1)} MB`);
  if (elapsed != null) parts.push(`analyzed in ${fmtClock(elapsed)}`);
  return (
    <div className="vid-strip">
      <span className="vid-name">{videoName}</span>
      {parts.map((p) => (
        <span key={p} className="vid-meta mono">
          {p}
        </span>
      ))}
    </div>
  );
}

function SimpleVerdict({ result, onTryAgain }) {
  const label = result?.verdict?.label ?? "UNCERTAIN";
  const p = PLAIN[label] ?? PLAIN.UNCERTAIN;
  const conf = result?.verdict?.confidence;
  const probReal = result?.stages?.quantum?.prob_real;
  const score = probReal ?? conf;

  return (
    <section className="simple-verdict" aria-label="verification result">
      <div className={`sv-badge sv-${p.tone}`} aria-hidden="true">
        {p.icon}
      </div>
      <div className="sv-main">
        <span className="sv-eyebrow">Verification result</span>
        <h2 className={`sv-word tone-${p.tone}`}>{p.word}</h2>
        <p className="sv-note">{p.note}</p>
        {conf != null && (
          <p className="sv-conf">
            {confWord(conf)}
            {score != null && (
              <span className="sv-conf-num mono"> · live-signal score {fmtVal(score, 3)}</span>
            )}
          </p>
        )}
        <div className="sv-actions">
          <button className="btn btn-primary" onClick={onTryAgain}>
            Analyze another video
          </button>
        </div>
      </div>
    </section>
  );
}

function Checklist({ result }) {
  const frames = result?.stages?.frames;
  const rppg = result?.stages?.rppg;
  const quantum = result?.stages?.quantum;
  const xc = result?.stages?.rppg_crosscheck;
  const label = result?.verdict?.label;

  const items = [];

  if (frames?.status === "success") {
    const a = frames.stats?.accepted_frames ?? 0;
    const s = frames.stats?.sampled_frames ?? 0;
    items.push({
      ok: a >= Math.max(1, s * 0.5) ? "ok" : "warn",
      title: "The video was clear enough to analyze",
      detail: `${a} of ${s} sampled frames passed the quality check`,
    });
  } else {
    items.push({
      ok: "skip",
      title: "Video quality check",
      detail: "Quality details are not available for this run",
    });
  }

  if (rppg?.features) {
    items.push({
      ok: "ok",
      title: "A pulse was measured from the face",
      detail: `≈ ${Math.round(rppg.features.heart_rate_bpm ?? 0)} beats per minute across ${rppg.n_frames_usable} frames`,
    });
  } else {
    items.push({
      ok: "bad",
      title: "No pulse could be measured from the face",
      detail: "The video may be too short, too dark, or the face was not clearly visible",
    });
  }

  if (quantum) {
    const pr = quantum.prob_real ?? 0.5;
    items.push({
      ok: label === "REAL" ? "ok" : label === "FAKE" ? "bad" : "warn",
      title:
        label === "REAL"
          ? "The pulse pattern matches a live recording"
          : label === "FAKE"
            ? "The pulse pattern does not match a live recording"
            : "The pulse pattern is too close to call",
      detail: `final score ${fmtVal(pr, 3)} — a score of 0.70 or higher points to a real person`,
    });
  }

  if (xc?.verdict) {
    items.push({
      ok: xc.verdict === "DEEPFAKE" ? "bad" : "ok",
      title:
        xc.verdict === "DEEPFAKE"
          ? "The machine-learning cross-check flagged the video"
          : "The machine-learning cross-check found no signs of tampering",
      detail: `independent model confidence ${fmtVal(xc.probability, 3)}`,
    });
  } else {
    items.push({
      ok: "skip",
      title: "Machine-learning cross-check",
      detail: "Not available for this run",
    });
  }

  return (
    <section className="card">
      <header className="card-head">
        <h3 className="card-title">What we checked</h3>
      </header>
      <div className="checklist-body">
        {items.map((it, i) => (
          <div key={i} className={`check-item check-${it.ok}`}>
            <span className="check-mark" aria-hidden="true">
              {it.ok === "ok" ? "✓" : it.ok === "bad" ? "✗" : it.ok === "warn" ? "!" : "–"}
            </span>
            <div>
              <p className="check-title">{it.title}</p>
              <p className="check-detail">{it.detail}</p>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

function SampleFrames({ videoStem }) {
  const thumbs = useThumbs(videoStem);
  if (!thumbs || thumbs.length === 0) return null;
  return (
    <section className="card card-wide">
      <header className="card-head">
        <h3 className="card-title">Sample frames from your video</h3>
      </header>
      <div className="card-body">
        <div className="samples">
          {thumbs.map((f) => (
            <img key={f.rel} src={fileUrl(f.rel)} alt={f.name} loading="lazy" />
          ))}
        </div>
        <p className="samples-cap mono">frames used for the analysis · {thumbs.length} of the stored frames shown</p>
      </div>
    </section>
  );
}

function FriendlyProgress({ stageIdx, elapsed, videoName }) {
  const active = Math.min(stageIdx, 2);
  const barValue = stageIdx === 0 ? null : Math.round((stageIdx / 3) * 100);
  return (
    <section className="bay bay-running" role="status" aria-live="polite">
      <div className="run-head">
        <span className="live-dot" aria-hidden="true" />
        <span className="fstatus-head">Analyzing… {fmtClock(elapsed)}</span>
      </div>
      <ProgressBar
        value={barValue}
        pendingLabel="Working"
        completeLabel="Analysis complete"
        ariaLabel={videoName ? `Checking ${videoName}` : "Analysis progress"}
        className="pb-inline"
      />
      <ol className="fsteps">
        {SIMPLE_STEPS.map(([title, sub], i) => (
          <li key={title} className={`fstep${i < active ? " done" : ""}${i === active ? " active" : ""}`}>
            <span className="fstep-mark" aria-hidden="true">
              {i < active ? "✓" : i === active ? "▸" : "·"}
            </span>
            <div>
              <p className="fstep-title">{title}</p>
              <p className="fstep-sub">{sub}</p>
            </div>
          </li>
        ))}
      </ol>
    </section>
  );
}

/* ------------------------------------------------------------------ */
/* upload bay + progress                                               */
/* ------------------------------------------------------------------ */

function UploadBay({ phase, stageIdx, elapsed, videoName, onFile }) {
  const [drag, setDrag] = useState(false);
  const inputRef = useRef(null);

  const pick = (files) => {
    const f = files?.[0];
    if (f && f.type.startsWith("video/")) onFile(f);
  };

  if (phase === "running") {
    return <FriendlyProgress stageIdx={stageIdx} elapsed={elapsed} videoName={videoName} />;
  }

  const isError = phase === "error";
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
      <h2 className="bay-title">
        {isError ? "Something went wrong" : "Upload a video to check if it's real or AI-generated"}
      </h2>
      <p className="bay-sub">
        {isError
          ? "The analysis server hit an error. Try again, or check the server console for details."
          : "Works best with short selfie-style or KYC videos. Everything runs on this machine — nothing is uploaded."}
      </p>
      <button className="btn btn-primary" onClick={() => inputRef.current?.click()}>
        {isError ? "Try again" : "Choose a video"}
      </button>
      <p className="bay-hint mono">or drag and drop a video here</p>
    </section>
  );
}

/* ------------------------------------------------------------------ */
/* app                                                                 */
/* ------------------------------------------------------------------ */

export default function App() {
  const [phase, setPhase] = useState("idle"); // idle | running | done | error
  const [result, setResult] = useState(null);
  const [stageIdx, setStageIdx] = useState(0);
  const [videoName, setVideoName] = useState(null);
  const [videoMeta, setVideoMeta] = useState(null);
  const [lastElapsed, setLastElapsed] = useState(null);
  const [runError, setRunError] = useState(null);
  const [theme, setTheme] = useTheme();
  const elapsed = useElapsed(phase === "running");
  const elapsedRef = useRef(0);
  elapsedRef.current = elapsed;

  useEffect(() => {
    previous().then(({ result: prev }) => {
      if (prev) {
        setResult(prev);
        setVideoName(prev.video);
      }
    });
  }, []);

  const reset = () => {
    setPhase("idle");
    setResult(null);
    setStageIdx(0);
    setVideoName(null);
    setVideoMeta(null);
    setLastElapsed(null);
    setRunError(null);
  };

  const run = useCallback((file) => {
    setPhase("running");
    setResult(null);
    setStageIdx(0);
    setVideoName(file.name);
    setVideoMeta({ size: file.size, duration: 0 });
    setLastElapsed(null);
    setRunError(null);
    try {
      const url = URL.createObjectURL(file);
      const v = document.createElement("video");
      v.preload = "metadata";
      v.onloadedmetadata = () => {
        setVideoMeta((m) => (m ? { ...m, duration: v.duration } : m));
        URL.revokeObjectURL(url);
      };
      v.onerror = () => URL.revokeObjectURL(url);
      v.src = url;
    } catch {
      /* duration is a nice-to-have; ignore failure */
    }
    detect(file)
      .then(({ job }) => {
        stream(job, {
          stage: (l) => {
            const m = l.match(/\[(\d)\/3\]/);
            if (m) setStageIdx(parseInt(m[1], 10));
          },
          result: (res) => {
            setResult(res);
            setStageIdx(3);
            setPhase("done");
            setLastElapsed(elapsedRef.current);
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

  const videoStem = result?.video ? result.video.replace(/\.\w+$/, "") : null;
  const haveResult = Boolean(result) && phase !== "running" && phase !== "error";

  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-glyph" aria-hidden="true">
            ∿
          </span>
          <div>
            <h1>
              rPPG·QC <span className="brand-amp">Verifier</span>
            </h1>
            <p>Deepfake verification for identity checks</p>
          </div>
        </div>
        <div className="topbar-right">
          <span className="ok-dot" aria-hidden="true" />
          Your video never leaves this machine
          <button
            type="button"
            className="theme-toggle"
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
          >
            {theme === "dark" ? "Light mode" : "Dark mode"}
          </button>
        </div>
      </header>

      <main>
        {phase === "error" && <div className="error-banner mono">{runError}</div>}

        <UploadBay
          phase={phase}
          stageIdx={stageIdx}
          elapsed={elapsed}
          videoName={videoName}
          onFile={run}
        />

        {haveResult && (
          <>
            <VideoStrip videoName={videoName} meta={videoMeta} elapsed={lastElapsed} />
            <SimpleVerdict result={result} onTryAgain={reset} />
            <Checklist result={result} />
            <SampleFrames videoStem={videoStem} />
          </>
        )}
      </main>

      <footer className="foot">
        <span>Verified locally — the video never leaves this machine.</span>
      </footer>
    </div>
  );
}
