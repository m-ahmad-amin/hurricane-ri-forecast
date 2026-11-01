export default function LogoMark({ className = "" }: { className?: string }) {
  return (
    <span className={`logo-mark ${className}`} aria-hidden="true">
      <svg viewBox="0 0 36 36" width="26" height="26" fill="none">
        <path
          d="M18 4c7.2 2.2 11.5 8.4 10.2 15.2-1 5.2-5.2 9-10.2 9.8"
          stroke="currentColor"
          strokeWidth="2.2"
          strokeLinecap="round"
        />
        <path
          d="M18 32C10.8 29.8 6.5 23.6 7.8 16.8c1-5.2 5.2-9 10.2-9.8"
          stroke="currentColor"
          strokeWidth="2.2"
          strokeLinecap="round"
        />
        <circle cx="18" cy="18" r="3.2" fill="currentColor" />
        <circle cx="18" cy="18" r="6.2" stroke="currentColor" strokeWidth="1.4" opacity="0.35" />
      </svg>
    </span>
  );
}
