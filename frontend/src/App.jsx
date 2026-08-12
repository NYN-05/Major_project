import { useCallback, useEffect, useRef, useState } from "react";
import { motion } from "motion/react";
import { AlertTriangle } from "lucide-react";
import { detect, fileUrl, previous, stream } from "./api.js";
import { useElapsed, useSignalFile, useTheme } from "./hooks.js";
import Header from "./components/Header.jsx";
import UploadZone from "./components/UploadZone.jsx";
import VideoSelected from "./components/VideoSelected.jsx";
import ProcessingView from "./components/ProcessingView.jsx";
import VerdictHeader from "./components/VerdictHeader.jsx";
import VerdictCard from "./components/VerdictCard.jsx";
import KeyMetrics from "./components/KeyMetrics.jsx";
import SignalPanel from "./components/SignalPanel.jsx";
import QuantumFlow from "./components/QuantumFlow.jsx";
import CrossCheck from "./components/CrossCheck.jsx";
import FrameGallery from "./components/FrameGallery.jsx";
import TechnicalDetails from "./components/TechnicalDetails.jsx";
import { artifacts } from "./api.js";

export default function App() {
  const [phase, setPhase] = useState("idle"); // idle | selected | running | done | error
  const [file, setFile] = useState(null);
  const [videoMeta, setVideoMeta] = useState(null); // local probe: {name,size,duration,width,height}
  const [result, setResult] = useState(null);
  const [stageIdx, setStageIdx] = useState(0);
  const [videoName, setVideoName] = useState(null);
  const [lastElapsed, setLastElapsed] = useState(null);
  const [runError, setRunError] = useState(null);
  const [lines, setLines] = useState([]);
  const [signalRel, setSignalRel] = useState(null);
  const [theme, setTheme] = useTheme();
  const elapsed = useElapsed(phase === "running");
  const elapsedRef = useRef(0);
  elapsedRef.current = elapsed;

  useEffect(() => {
    previous().then(({ result: prev }) => {
      if (prev) {
        setResult(prev);
        setVideoName(prev.video?.name ?? prev.video);
        setPhase("done");
      }
    });
  }, []);

  const reset = () => {
    setPhase("idle");
    setFile(null);
    setVideoMeta(null);
    setResult(null);
    setStageIdx(0);
    setVideoName(null);
    setLastElapsed(null);
    setRunError(null);
    setLines([]);
    setSignalRel(null);
  };

  const pickFile = (f) => {
    setRunError(null);
    setFile(f);
    setSignalRel(null);
    const meta = { name: f.name, size: f.size, duration: null, width: null, height: null };
    try {
      const url = URL.createObjectURL(f);
      const v = document.createElement("video");
      v.preload = "metadata";
      v.onloadedmetadata = () => {
        setVideoMeta({
          name: f.name,
          size: f.size,
          duration: Number.isFinite(v.duration) ? v.duration : null,
          width: v.videoWidth || null,
          height: v.videoHeight || null,
        });
        URL.revokeObjectURL(url);
      };
      v.onerror = () => URL.revokeObjectURL(url);
      v.src = url;
    } catch {
      /* metadata is best-effort */
    }
    setPhase("selected");
  };

  const run = useCallback(() => {
    if (!file) return;
    setPhase("running");
    setResult(null);
    setStageIdx(0);
    setVideoName(file.name);
    setLastElapsed(null);
    setRunError(null);
    setLines([]);
    setSignalRel(null);
    detect(file)
      .then(({ job }) => {
        stream(job, {
          line: (l) => setLines((prev) => [...prev, l].slice(-40)),
          stage: (l) => {
            const m = l.match(/\[(\d)\/3\]/);
            if (m) setStageIdx(parseInt(m[1], 10));
          },
          signal: (rel) => setSignalRel(rel),
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
  }, [file]);

  const signalRel2 = result?._signal ?? signalRel;
  const signalData = useSignalFile(fileUrl, signalRel2);

  return (
    <div className="shell">
      <Header theme={theme} onToggleTheme={() => setTheme(theme === "dark" ? "light" : "dark")} />

      <main>
        {phase === "error" && (
          <motion.div
            className="error-banner"
            role="alert"
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <AlertTriangle size={15} aria-hidden="true" />
            <span>{runError}</span>
          </motion.div>
        )}

        {phase === "selected" ? (
          <VideoSelected
            file={file}
            meta={videoMeta}
            onStart={run}
            onChange={() => setPhase("idle")}
          />
        ) : phase === "running" ? (
          <ProcessingView
            stageIdx={stageIdx}
            elapsed={elapsed}
            videoName={videoName}
            lines={lines}
          />
        ) : phase === "done" ? (
          <div className="results">
            <VerdictHeader result={result} videoMeta={videoMeta} resultElapsed={lastElapsed} />
            <div className="result-top">
              <VerdictCard result={result} onTryAgain={reset} />
              <KeyMetrics result={result} />
            </div>
            <div className="row-2">
              <SignalPanel signalData={signalData} result={result} />
              <QuantumFlow result={result} />
            </div>
            <CrossCheck result={result} />
            <FrameGallery result={result} artifacts={artifacts} />
            <TechnicalDetails result={result} videoMeta={videoMeta} />
          </div>
        ) : (
          <UploadZone phase={phase} onFile={pickFile} />
        )}
      </main>

      <footer className="foot">
        <span>Verified locally — the video never leaves this machine</span>
      </footer>
    </div>
  );
}