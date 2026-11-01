import { useEffect, useState } from "react";

const STEPS = [
  "Connecting to the forecast service",
  "Matching similar storms in the archive",
  "Comparing wind, basin, and location",
  "Averaging calibrated RI probabilities",
  "Preparing your 24-hour RI chance",
];

export default function EstimateProgress({ active }: { active: boolean }) {
  const [step, setStep] = useState(0);

  useEffect(() => {
    if (!active) {
      setStep(0);
      return;
    }
    setStep(0);
    const delays = [0, 1800, 4200, 7200, 11000];
    const timers = delays.map((ms, i) => window.setTimeout(() => setStep(i), ms));
    return () => timers.forEach((t) => window.clearTimeout(t));
  }, [active]);

  if (!active) return null;

  return (
    <div className="estimate-progress">
      <p className="estimate-progress-note">
        Matching archive storms. The first request can take longer if the service is waking up.
      </p>
      <ol className="estimate-steps">
        {STEPS.map((label, i) => {
          const state = i < step ? "done" : i === step ? "active" : "todo";
          return (
            <li key={label} className={`estimate-step ${state}`}>
              <span className="estimate-dot" aria-hidden>
                {state === "active" ? <span className="estimate-spin" /> : null}
                {state === "done" ? (
                  <svg viewBox="0 0 16 16" width="10" height="10" aria-hidden>
                    <path
                      d="M3.5 8.5 6.5 11.5 12.5 4.5"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                ) : null}
              </span>
              <span className="estimate-label">{label}</span>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
