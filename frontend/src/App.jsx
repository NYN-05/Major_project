import { useCallback, useEffect, useRef, useState } from "react";
import { motion } from "motion/react";
import { artifacts, detect, fileUrl, previous, stream } from "./api.js";
import { useElapsed, useSignalFile, useTheme } from "./hooks.js";
import Header from "./components/Header.jsx";
import UploadZone from "./components/UploadZone.jsx";
import Pipeline from "./components/Pipeline.jsx";
import VerdictPanel from "./components/VerdictPanel.jsx";
import Insights from "./components/Insights.jsx";
import SignalViz from "./components/SignalViz.jsx";
import QuantumViz from "./components/QuantumViz.jsx";
import FileInfo from "./components/FileInfo.jsx";
import FrameSamples from "./components/FrameSamples.jsx";

const stemOf = (name) => (name ? String(name).replace(/\.\w+$/, "") : null);

export default function App() {
  const [phase, setPhase] = useState("idle"); // idle | running | done | error
  const [result, setResult] = useState(null);
  const [stageIdx, setStageIdx] = useState(0);
  const [videoName, setVideoName] = useState(null);
  const [lastElapsed, setLastElapsed] = useState(null);
  const [runError, setRunError] = useState(null);
  const [lines, setLines] = useState([]);
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
    setLastElapsed(null);
    setRunError(null);
    setLines([]);
  };

  const run = useCallback((file) => {
    setPhase("running");
    setResult(null);
    setStageIdx(0);
    setVideoName(file.name);
    setLastElapsed(null);
    setRunError(null);
    setLines([]);
    detect(file)
      .then(({ job }) => {
        stream(job, {
          line: (l) => setLines((prev) => [...prev, l].slice(-40)),
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

  const signalRel = result?._signal;
  const signalData = useSignalFile(fileUrl, signalRel);

  const haveResult = Boolean(result) && phase !== "running" && phase !== "error";
  const stem = stemOf(videoName ?? result?.video);

  return (
    <div className="shell">
      <div className="bg" aria-hidden="true">
        <div className="bg-grid" />
        <div className="bg-orb bg-orb-a" />
        <div className="bg-orb bg-orb-b" />
        <div className="bg-orb bg-orb-c" />
      </div>

      <Header theme={theme} onToggleTheme={() => setTheme(theme === "dark" ? "light" : "dark")} />

      <main>
        {phase === "error" && (
          <motion.div
            className="error-banner mono"
            role="alert"
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
          >
            {runError}
          </motion.div>
        )}

        {phase === "running" || phase === "done" ? (
          <Pipeline
            stageIdx={stageIdx}
            elapsed={elapsed}
            videoName={videoName}
            lines={lines}
            phase={phase}
            result={result}
            resultElapsed={lastElapsed}
          />
        ) : (
          <UploadZone phase={phase} onFile={run} />
        )}

        {haveResult && (
          <>
            <VerdictPanel result={result} onTryAgain={reset} />
            <Insights result={result} />
            <SignalViz signalData={signalData} bpm={result?.stages?.rppg?.features?.heart_rate_bpm} sqi={result?.stages?.rppg?.features?.signal_quality_index} />
            <QuantumViz result={result} />
            <FileInfo result={result} />
            <FrameSamples result={result} artifacts={artifacts} />
          </>
        )}
      </main>

      <footer className="foot">
        <span>Verified locally — the video never leaves this machine.</span>
      </footer>
    </div>
  );
}