import { useEffect, useRef } from "react";
import { useT } from "../theme/ThemeContext";
import { useViewport } from "../hooks/useViewport";
import { DESKTOP_BUILDS, DESKTOP_RELEASES_URL } from "../data/links";
import type { DesktopOS } from "../utils/platform";
import { OsIcon } from "./OsIcon";
import { MacInstallHelp } from "./MacInstallHelp";

const FOCUSABLE = 'a[href],button:not([disabled]),input,select,textarea,[tabindex]:not([tabindex="-1"])';

/**
 * Modal listing every desktop build (macOS / Windows / Linux) with its file
 * name and architecture, the macOS first-launch steps, and a link to all
 * releases. The visitor's detected OS is highlighted. Escape, the backdrop,
 * and the close button dismiss it; focus is trapped inside while open and
 * returned to the opener on close.
 */
export function DownloadPanel({ detected, onClose }: { detected: DesktopOS; onClose: () => void }) {
  const t = useT();
  const { compact } = useViewport();
  const dialogRef = useRef<HTMLDivElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);
  // Latest onClose without re-running the mount effect (which moves focus).
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;

  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null;
    closeRef.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") { e.preventDefault(); onCloseRef.current(); return; }
      if (e.key !== "Tab" || !dialogRef.current) return;
      const els = Array.from(dialogRef.current.querySelectorAll<HTMLElement>(FOCUSABLE));
      if (els.length === 0) return;
      const first = els[0], last = els[els.length - 1];
      if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
      else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
    };
    document.addEventListener("keydown", onKey);
    return () => { document.removeEventListener("keydown", onKey); opener?.focus?.(); };
  }, []);

  const pad = compact ? 14 : 20;
  return (
    <div onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
      style={{ position: "fixed", inset: 0, zIndex: 1100, background: "rgba(0,0,0,0.55)", display: "flex", alignItems: "center", justifyContent: "center", padding: 16 }}>
      <div ref={dialogRef} role="dialog" aria-modal="true" aria-labelledby="mv-dl-title"
        style={{ width: "min(520px, 100%)", maxHeight: "calc(100dvh - 32px)", overflowY: "auto", background: t.bg1, color: t.t0, border: `1px solid ${t.bdr}`, borderRadius: 12, padding: pad, boxShadow: "0 18px 50px rgba(0,0,0,0.35)", fontFamily: "'Inter',-apple-system,sans-serif" }}>
        <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 12, marginBottom: 12 }}>
          <div style={{ minWidth: 0 }}>
            <div id="mv-dl-title" style={{ fontSize: 15, fontWeight: 700, fontFamily: "'Space Grotesk'", color: t.t0 }}>Download ModelVisio Desktop</div>
            <div style={{ fontSize: 11, color: t.t2, marginTop: 2 }}>Native app for macOS, Windows and Linux · always the latest release</div>
          </div>
          <button ref={closeRef} type="button" onClick={onClose} aria-label="Close download panel" title="Close"
            style={{ width: 28, height: 28, flexShrink: 0, borderRadius: 6, border: `1px solid ${t.bdr}`, background: t.bg2, color: t.t1, fontSize: 12, cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center" }}>✕</button>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {DESKTOP_BUILDS.map((b) => {
            const mine = b.os === detected;
            return (
              <section key={b.os} aria-label={`${b.label} download`}
                style={{ border: `1px solid ${mine ? t.acc + "88" : t.bdr}`, background: mine ? t.acc + "0d" : t.bg, borderRadius: 9, padding: compact ? "10px 11px" : "11px 13px" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
                  <span style={{ color: t.t1, display: "inline-flex" }}><OsIcon os={b.os} size={18} /></span>
                  <div style={{ flex: "1 1 180px", minWidth: 0 }}>
                    <div style={{ fontSize: 12.5, fontWeight: 600, color: t.t0, display: "flex", alignItems: "center", gap: 6 }}>
                      {b.label}
                      {mine && <span style={{ fontSize: 9, fontWeight: 700, letterSpacing: .5, textTransform: "uppercase", color: t.acc, border: `1px solid ${t.acc}55`, borderRadius: 999, padding: "0 6px" }}>Your system</span>}
                    </div>
                    <div style={{ fontSize: 10.5, color: t.t2 }}>{b.arch}</div>
                    <div style={{ fontSize: 10, color: t.t3, fontFamily: "'JetBrains Mono',monospace", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{b.file}</div>
                  </div>
                  <a href={b.url} target="_blank" rel="noreferrer noopener" aria-label={`Download ModelVisio for ${b.label} (${b.file})`}
                    style={{ padding: "6px 14px", borderRadius: 7, background: mine ? t.acc : "transparent", border: `1px solid ${mine ? t.acc : t.bdr}`, color: mine ? "#fff" : t.t1, fontSize: 11.5, fontWeight: 600, textDecoration: "none", whiteSpace: "nowrap", marginLeft: compact ? 28 : 0 }}>
                    Download
                  </a>
                </div>
                {b.os === "mac" && (
                  <div style={{ marginTop: 9, paddingTop: 9, borderTop: `1px solid ${t.bdr}` }}>
                    <MacInstallHelp />
                  </div>
                )}
              </section>
            );
          })}
        </div>

        <div style={{ marginTop: 12, fontSize: 11, color: t.t2, textAlign: "center" }}>
          Looking for .msi, .deb or .rpm?{" "}
          <a href={DESKTOP_RELEASES_URL} target="_blank" rel="noreferrer noopener" style={{ color: t.acc, fontWeight: 600, textDecoration: "none" }}>All releases →</a>
        </div>
      </div>
    </div>
  );
}
