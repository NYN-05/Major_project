import { motion } from "motion/react";
import { Lock, Moon, Sun } from "lucide-react";

export default function Header({ theme, onToggleTheme }) {
  return (
    <motion.header
      className="topbar"
      initial={{ opacity: 0, y: -12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
    >
      <div className="brand">
        <div className="brand-tile" aria-hidden="true">
          <svg className="brand-wave" viewBox="0 0 34 22" fill="none">
            <path
              d="M1 12 C4 3, 7 21, 10 12 S16 3, 19 12 S25 21, 28 12 S31 4, 33 10"
              stroke="currentColor"
              strokeWidth="1.7"
              strokeLinecap="round"
            />
          </svg>
        </div>
        <div className="brand-text">
          <h1>
            rPPG·QC <span className="brand-amp">Verifier</span>
          </h1>
          <p>Deepfake verification for KYC</p>
        </div>
      </div>

      <div className="topbar-actions">
        <div className="sec-badge" role="status">
          <span className="sec-dot" aria-hidden="true" />
          <Lock size={11} aria-hidden="true" />
          <span>Video stays on this machine</span>
        </div>
        <button
          type="button"
          className="icon-btn theme-toggle"
          onClick={onToggleTheme}
          aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
          title={theme === "dark" ? "Light mode" : "Dark mode"}
        >
          <motion.span
            key={theme}
            initial={{ rotate: -90, opacity: 0, scale: 0.6 }}
            animate={{ rotate: 0, opacity: 1, scale: 1 }}
            transition={{ duration: 0.3, ease: "easeOut" }}
            className="theme-toggle-ico"
          >
            {theme === "dark" ? <Sun size={15} /> : <Moon size={15} />}
          </motion.span>
        </button>
      </div>

      <div className="topbar-rule" aria-hidden="true" />
    </motion.header>
  );
}