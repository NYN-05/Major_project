import { useRef, useState } from "react";
import { motion } from "motion/react";
import { Upload } from "lucide-react";

export default function UploadZone({ phase, onFile }) {
  const [drag, setDrag] = useState(false);
  const inputRef = useRef(null);
  const isError = phase === "error";

  const pick = (files) => {
    const f = files?.[0];
    if (f && f.type.startsWith("video/")) onFile(f);
  };

  return (
    <motion.section
      className={`upload${drag ? " upload-drag" : ""}`}
      initial={{ opacity: 0, y: 18 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1], delay: 0.08 }}
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
      role="button"
      tabIndex={0}
      aria-label="Upload a video for deepfake verification"
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          inputRef.current?.click();
        }
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept="video/*"
        hidden
        onChange={(e) => pick(e.target.files)}
      />
      <div className="upload-border" aria-hidden="true" />

      <div className="upload-core" aria-hidden="true">
        <span className="pulse-ring pulse-ring-1" />
        <span className="pulse-ring pulse-ring-2" />
        <span className="pulse-ring pulse-ring-3" />
        <div className="upload-glyph">
          <svg className="upload-wave" viewBox="0 0 120 40" fill="none">
            <path
              d="M2 20 C 10 6, 18 34, 26 20 S 42 6, 50 20 S 66 34, 74 20 S 90 6, 98 20 S 112 28, 118 18"
              stroke="currentColor"
              strokeWidth="1.8"
              strokeLinecap="round"
            />
          </svg>
          <span className="upload-core-dot" />
          <span className="scan-line" />
        </div>
      </div>

      <motion.h2
        className="upload-title"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2, duration: 0.5 }}
      >
        {isError ? "Something went wrong" : "Verify a video is real or AI-generated"}
      </motion.h2>
      <p className="upload-sub">
        {isError
          ? "The analysis server hit an error on the last run. Try again, or check the server console."
          : "Upload a short selfie-style or KYC video. The system measures the subtle pulse of the face to decide."}
      </p>

      <div className="upload-actions">
        <motion.button
          type="button"
          className="btn btn-primary"
          onClick={() => inputRef.current?.click()}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.97 }}
        >
          <Upload size={15} aria-hidden="true" />
          {isError ? "Try again" : "Choose a video"}
        </motion.button>
      </div>

      <p className="upload-hint">or drag and drop a video here</p>
      <p className="upload-fmts mono">
        MP4 · AVI · MOV <span className="upload-fmts-dot" /> samples ~10 fps during analysis
      </p>
    </motion.section>
  );
}