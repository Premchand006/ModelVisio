// Visitor OS detection for the landing page's desktop download button. Pure:
// takes a navigator-like object so it is trivially testable, and is SSR-safe
// (no browser `window` → "other").

/** Desktop OS we ship an installer for, or "other" (mobile, unknown, SSR). */
export type DesktopOS = "mac" | "windows" | "linux" | "other";

/** The subset of `Navigator` that detectOS reads. */
export type NavigatorLike = {
  userAgentData?: { platform?: string; mobile?: boolean } | null;
  platform?: string;
  userAgent?: string;
  maxTouchPoints?: number;
};

const MOBILE_UA = /android|iphone|ipad|ipod|mobile|silk|kindle|blackberry|iemobile|opera mini/i;

function classify(s: string): DesktopOS | null {
  const v = s.toLowerCase();
  if (!v) return null;
  // Mobile platforms first: Android reports "Linux armv8l" etc. in navigator.platform.
  // ChromeOS also says "Linux"/"X11" but can't run our installers directly.
  if (/android|iphone|ipad|ipod|\bios\b|cros|chrome ?os/.test(v)) return "other";
  if (/mac|darwin/.test(v)) return "mac";
  if (/win/.test(v)) return "windows";
  if (/linux|x11|ubuntu|fedora|debian|bsd/.test(v)) return "linux";
  return null;
}

/**
 * Best-effort desktop OS of the visitor. Checks `userAgentData.platform`
 * (Chromium), then the legacy `navigator.platform`, then the user-agent string.
 * Phones and tablets map to "other", including iPadOS Safari, which pretends
 * to be a Mac ("MacIntel") but exposes multi-touch.
 *
 * With no argument it reads `window.navigator`; in a non-browser environment
 * (SSR, or Node, whose global `navigator` describes the server) it returns
 * "other".
 */
export function detectOS(nav?: NavigatorLike | null): DesktopOS {
  const n: NavigatorLike | undefined =
    nav ?? (typeof window !== "undefined" ? (window.navigator as NavigatorLike | undefined) : undefined);
  if (!n) return "other";

  const ua = n.userAgent ?? "";
  if (n.userAgentData?.mobile || MOBILE_UA.test(ua)) return "other";

  const os =
    classify(n.userAgentData?.platform ?? "") ??
    classify(n.platform ?? "") ??
    classify(ua) ??
    "other";

  // iPadOS 13+ Safari reports a Mac platform; real Macs have no touch screen.
  if (os === "mac" && (n.maxTouchPoints ?? 0) > 1) return "other";
  return os;
}
