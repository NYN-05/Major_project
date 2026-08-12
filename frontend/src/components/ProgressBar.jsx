import { motion, useReducedMotion } from "motion/react";

const FILL = { type: "spring", stiffness: 210, damping: 34, mass: 0.9 };
const CROSSFADE = { type: "spring", stiffness: 260, damping: 34, mass: 0.8 };
const INSTANT = { duration: 0 };

export default function ProgressBar({
  value,
  max = 100,
  pendingLabel = "Working",
  completeLabel = "Complete",
  ariaLabel = "Progress",
  className = "",
}) {
  const reduced = useReducedMotion();

  const indeterminate = value === null;
  const fraction = value === null || max <= 0 ? 0 : Math.min(1, Math.max(0, value / max));
  const percent = Math.round(fraction * 100);
  const complete = !indeterminate && fraction >= 1;

  const measured = indeterminate
    ? {}
    : {
        "aria-valuenow": Math.round(fraction * max * 100) / 100,
        "aria-valuetext": `${percent}%`,
      };

  return (
    <div className={`progress-bar ${className}`}>
      <div className="pb-value" aria-hidden="true">
        <motion.span
          className="pb-pending"
          initial={false}
          animate={{ opacity: indeterminate ? 1 : 0 }}
          transition={reduced ? INSTANT : CROSSFADE}
        >
          {pendingLabel}
        </motion.span>
        <motion.span
          className="pb-percent"
          initial={false}
          animate={{ opacity: indeterminate ? 0 : 1 }}
          transition={reduced ? INSTANT : CROSSFADE}
        >
          {percent}%
        </motion.span>
      </div>

      <div
        role="progressbar"
        aria-label={ariaLabel}
        aria-valuemin={0}
        aria-valuemax={max}
        {...measured}
        className="pb-track"
      >
        <div className="pb-inner">
          <motion.span
            aria-hidden
            className="pb-fill"
            initial={false}
            animate={{ scaleX: indeterminate ? 0 : fraction }}
            transition={reduced ? INSTANT : FILL}
          />
          {indeterminate && !reduced && (
            <motion.span
              aria-hidden
              className="pb-shimmer"
              initial={{ x: "-100%", opacity: 0 }}
              animate={{ x: "250%", opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{
                x: { duration: 1.25, ease: "easeInOut", repeat: Infinity },
                opacity: { duration: 0.18 },
              }}
            />
          )}
          {!indeterminate && !reduced && (
            <motion.span
              aria-hidden
              className="pb-sheen"
              initial={{ x: "-60%" }}
              animate={{ x: "160%" }}
              transition={{ duration: 1.6, ease: "easeInOut", repeat: Infinity }}
            />
          )}
        </div>
      </div>

      <span aria-live="polite" className="sr-only">
        {complete ? completeLabel : indeterminate ? pendingLabel : ""}
      </span>
    </div>
  );
}
