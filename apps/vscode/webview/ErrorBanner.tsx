import { TH, type ThemeName } from "@modelvisio/core";

// Shown when the host couldn't read the model file. Floats over the app (which
// stays usable — e.g. drag-and-drop another file) and can be dismissed.
export function ErrorBanner({ theme, title, message, onDismiss }: { theme: ThemeName; title: string; message: string; onDismiss: () => void }) {
  const T = TH[theme];
  return (
    <div
      role="alert"
      style={{
        position: "fixed", top: 12, left: "50%", transform: "translateX(-50%)", zIndex: 10000,
        maxWidth: "min(640px, calc(100vw - 32px))", display: "flex", alignItems: "flex-start", gap: 12,
        padding: "10px 12px", borderRadius: 8, background: T.bg1, color: T.t0,
        border: `1px solid ${T.bdr}`, borderLeft: `3px solid ${T.err}`,
        boxShadow: "0 6px 24px rgba(0,0,0,0.25)", font: "13px/1.45 system-ui, sans-serif",
      }}
    >
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontWeight: 600, color: T.err, marginBottom: 2 }}>{title}</div>
        <div style={{ color: T.t1, overflowWrap: "anywhere" }}>{message}</div>
      </div>
      <button
        type="button"
        aria-label="Dismiss"
        onClick={onDismiss}
        style={{ background: "none", border: "none", color: T.t2, cursor: "pointer", fontSize: 16, lineHeight: 1, padding: 2 }}
      >
        ×
      </button>
    </div>
  );
}
