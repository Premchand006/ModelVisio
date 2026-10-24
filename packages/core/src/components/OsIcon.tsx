import type { DesktopOS } from "../utils/platform";

// Small single-color OS glyphs (currentColor) for the desktop download UI,
// same pattern as the GitHub mark in App.tsx. "other" is a plain download arrow.
const APPLE_BODY = "M11.2 8.5c0-1.5 1.2-2.2 1.3-2.3-.7-1-1.8-1.2-2.2-1.2-.9-.1-1.8.6-2.3.6-.5 0-1.2-.5-2-.5-1 0-2 .6-2.5 1.6-1.1 1.9-.3 4.6.8 6.1.5.7 1.1 1.6 1.9 1.5.8 0 1-.5 2-.5s1.2.5 2 .5c.8 0 1.3-.8 1.8-1.5.6-.8.8-1.6.8-1.7 0 0-1.6-.6-1.6-2.6z";
const APPLE_LEAF = "M9.7 3.9c.4-.5.7-1.2.6-1.9-.6 0-1.3.4-1.8.9-.4.4-.7 1.1-.6 1.8.7.1 1.4-.3 1.8-.8z";
// Penguin: head with eye holes, body with a belly hole (even-odd), two feet.
const TUX_HEAD = "M5.1 4.3a2.9 2.9 0 1 0 5.8 0a2.9 2.9 0 1 0 -5.8 0zM6.4 3.9a.6.6 0 1 0 1.2 0a.6.6 0 1 0 -1.2 0zM8.4 3.9a.6.6 0 1 0 1.2 0a.6.6 0 1 0 -1.2 0z";
const TUX_BODY = "M3.4 10.2a4.6 5.2 0 1 0 9.2 0a4.6 5.2 0 1 0 -9.2 0zM5.3 11.1a2.7 3.5 0 1 0 5.4 0a2.7 3.5 0 1 0 -5.4 0z";
const TUX_FEET = "M3.8 15a1.8.9 0 1 0 3.6 0a1.8.9 0 1 0 -3.6 0zM8.6 15a1.8.9 0 1 0 3.6 0a1.8.9 0 1 0 -3.6 0z";
const ARROW = "M8 1.5v8.2M4.6 6.6 8 10l3.4-3.4M2.5 13.5h11";

export function OsIcon({ os, size = 14 }: { os: DesktopOS; size?: number }) {
  const common = { width: size, height: size, viewBox: "0 0 16 16", "aria-hidden": true as const, style: { flexShrink: 0 } };
  if (os === "mac") return <svg {...common} fill="currentColor"><path d={APPLE_BODY} /><path d={APPLE_LEAF} /></svg>;
  if (os === "windows") {
    return <svg {...common} fill="currentColor">
      <rect x="1.5" y="1.5" width="6" height="6" rx=".4" /><rect x="8.5" y="1.5" width="6" height="6" rx=".4" />
      <rect x="1.5" y="8.5" width="6" height="6" rx=".4" /><rect x="8.5" y="8.5" width="6" height="6" rx=".4" />
    </svg>;
  }
  if (os === "linux") {
    return <svg {...common} fill="currentColor">
      <path d={TUX_HEAD} fillRule="evenodd" /><path d={TUX_BODY} fillRule="evenodd" /><path d={TUX_FEET} />
    </svg>;
  }
  return <svg {...common} fill="none" stroke="currentColor" strokeWidth={1.6} strokeLinecap="round" strokeLinejoin="round"><path d={ARROW} /></svg>;
}
