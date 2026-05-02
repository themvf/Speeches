"use client";

export function BookmarkButton({
  saved,
  onToggle,
  size = 14,
}: {
  saved: boolean;
  onToggle: () => void;
  size?: number;
}) {
  return (
    <button
      type="button"
      aria-label={saved ? "Remove from saved" : "Save item"}
      aria-pressed={saved}
      onClick={(e) => {
        e.stopPropagation();
        onToggle();
      }}
      title={saved ? "Remove from saved" : "Save"}
      style={{
        background: "none",
        border: "none",
        cursor: "pointer",
        padding: "2px 4px",
        color: saved ? "#4fd5ff" : "rgba(139,149,161,0.5)",
        flexShrink: 0,
        lineHeight: 1,
        transition: "color 0.15s",
      }}
    >
      <svg
        width={size}
        height={size}
        viewBox="0 0 24 24"
        fill={saved ? "currentColor" : "none"}
        stroke="currentColor"
        strokeWidth={2}
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z" />
      </svg>
    </button>
  );
}
