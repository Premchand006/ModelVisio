// Small stroke icon set (24×24 paths) for the app chrome. Icons inherit the
// text color, so they follow the theme.

const PATHS = {
  analyze: "M17 10.5a6.5 6.5 0 1 1-13 0a6.5 6.5 0 0 1 13 0zM20 20l-4.6-4.6",
  optimize: "M13 2L4 14h7l-1 8 9-12h-7z",
  ship: "M12 15V3M7 8l5-5 5 5M5 21h14",
  copilot: "M21 12a8 8 0 0 1-11.8 7L4 20l1-4.6A8 8 0 1 1 21 12z",
  upload: "M12 15V3M7 8l5-5 5 5M4 15v4a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-4",
  warn: "M12 3l9.5 17h-19zM12 10v4M12 17.5v.5",
  close: "M6 6l12 12M18 6L6 18",
  sun: "M12 8a4 4 0 1 0 0 8a4 4 0 0 0 0-8zM12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4",
  moon: "M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z",
  layer: "M12 3l9 5-9 5-9-5zM3 13l9 5 9-5",
  action: "M5 12h14M13 6l6 6-6 6",
} as const;

export type IconName = keyof typeof PATHS;

export function Icon({ name, size = 16, stroke = 1.8 }: { name: IconName; size?: number; stroke?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={stroke}
      strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" style={{ flexShrink: 0 }}>
      <path d={PATHS[name]} />
    </svg>
  );
}
