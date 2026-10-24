import { useEffect, useRef, useState } from "react";
import { useT } from "../theme/ThemeContext";
import { MAC_INSTALL_GUIDE_URL } from "../data/links";

/** Terminal alternative to "Open Anyway": strips the download quarantine flag. */
export const MAC_UNQUARANTINE_CMD = "xattr -dr com.apple.quarantine /Applications/ModelVisio.app";

/**
 * First-launch steps for the macOS build. It isn't notarized, so Gatekeeper
 * blocks the first open; this explains the one-time approval plus a copyable
 * Terminal alternative. Shown in the download panel's macOS row and inline
 * under the landing-page button after a Mac visitor clicks Download.
 */
export function MacInstallHelp({ onDismiss }: { onDismiss?: () => void }) {
  const t = useT();
  const [copied, setCopied] = useState(false);
  const codeRef = useRef<HTMLElement>(null);
  const timer = useRef<ReturnType<typeof setTimeout>>();
  useEffect(() => () => clearTimeout(timer.current), []);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(MAC_UNQUARANTINE_CMD);
      setCopied(true);
      clearTimeout(timer.current);
      timer.current = setTimeout(() => setCopied(false), 1600);
    } catch {
      // Clipboard API unavailable/denied; select the command so ⌘C works.
      const el = codeRef.current;
      const sel = typeof window !== "undefined" ? window.getSelection() : null;
      if (el && sel) { const r = document.createRange(); r.selectNodeContents(el); sel.removeAllRanges(); sel.addRange(r); }
    }
  };

  return (
    <div style={{ textAlign: "left", fontSize: 11, lineHeight: 1.55, color: t.t2 }}>
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 8, marginBottom: 4 }}>
        <span style={{ fontWeight: 600, color: t.t1 }}>Installing on macOS</span>
        {onDismiss && <button type="button" onClick={onDismiss} aria-label="Dismiss macOS install steps" title="Dismiss"
          style={{ border: "none", background: "transparent", color: t.t3, fontSize: 11, lineHeight: 1, padding: 2, cursor: "pointer" }}>✕</button>}
      </div>
      <ol style={{ paddingLeft: 16, display: "flex", flexDirection: "column", gap: 2 }}>
        <li>Open the .dmg and drag ModelVisio to Applications.</li>
        <li>First launch: open it once, then go to <b style={{ color: t.t1, fontWeight: 600 }}>System Settings → Privacy &amp; Security</b> and click <b style={{ color: t.t1, fontWeight: 600 }}>"Open Anyway"</b> (macOS 15+).</li>
        <li>On older macOS you can Control-click the app → <b style={{ color: t.t1, fontWeight: 600 }}>Open</b>.</li>
      </ol>
      <div style={{ marginTop: 6, color: t.t3 }}>Or in Terminal:</div>
      <div style={{ marginTop: 3, display: "flex", alignItems: "stretch", border: `1px solid ${t.bdr}`, borderRadius: 6, background: t.bg2, overflow: "hidden" }}>
        <code ref={codeRef} style={{ flex: 1, minWidth: 0, padding: "5px 8px", fontFamily: "'JetBrains Mono',monospace", fontSize: 10.5, color: t.t1, overflowX: "auto", whiteSpace: "nowrap", userSelect: "all" }}>{MAC_UNQUARANTINE_CMD}</code>
        <button type="button" onClick={copy} aria-label="Copy Terminal command"
          style={{ flexShrink: 0, padding: "0 10px", border: "none", borderLeft: `1px solid ${t.bdr}`, background: "transparent", color: copied ? t.suc : t.t2, fontSize: 10.5, fontWeight: 600, cursor: "pointer" }}>
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <div style={{ marginTop: 5, fontSize: 10, color: t.t3 }}>
        The app isn't notarized by Apple yet, so Gatekeeper blocks the first launch only.{" "}
        <a href={MAC_INSTALL_GUIDE_URL} target="_blank" rel="noreferrer noopener" style={{ color: t.acc, fontWeight: 600, textDecoration: "none", whiteSpace: "nowrap" }}>Full install guide →</a>
      </div>
    </div>
  );
}
