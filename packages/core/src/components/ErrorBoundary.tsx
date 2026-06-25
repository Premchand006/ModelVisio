import { Component, type ErrorInfo, type ReactNode } from "react";
import { ThemeCtx } from "../theme/ThemeContext";

type Props = { children: ReactNode };
type State = { error: Error | null };

/**
 * Catches render-time crashes so a bug in one panel shows a recoverable message
 * instead of a blank white screen. Wraps the app tree in <App>, so all three
 * shells (web, desktop, VS Code) get the same safety net.
 */
export class ErrorBoundary extends Component<Props, State> {
  static contextType = ThemeCtx;
  declare context: React.ContextType<typeof ThemeCtx>;
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // Surface for debugging; the UI already shows a friendly fallback.
    console.error("ModelVisio crashed:", error, info.componentStack);
  }

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;
    const t = this.context;
    const bg = t?.bg ?? "#0A0A0B";
    const card = t?.bg1 ?? "#141417";
    const bdr = t?.bdr ?? "#27272E";
    const t0 = t?.t0 ?? "#F4F4F5";
    const t2 = t?.t2 ?? "#9A9AA4";
    const acc = t?.acc ?? "#4D8DFF";
    const err = t?.err ?? "#F87171";
    return (
      <div style={{ height: "100%", minHeight: 240, display: "flex", alignItems: "center", justifyContent: "center", padding: 24, background: bg, fontFamily: "'Inter',-apple-system,sans-serif" }}>
        <div style={{ maxWidth: 440, width: "100%", background: card, border: `1px solid ${bdr}`, borderRadius: 12, padding: 24, textAlign: "center" }}>
          <div style={{ fontSize: 30, marginBottom: 8 }}>⚠️</div>
          <div style={{ fontSize: 16, fontWeight: 700, color: t0, marginBottom: 6 }}>Something went wrong</div>
          <div style={{ fontSize: 12, color: t2, lineHeight: 1.6, marginBottom: 14 }}>
            A panel hit an unexpected error. Your data is safe — try again, or reload and re-open the model.
          </div>
          <pre style={{ fontSize: 10, color: err, background: bg, border: `1px solid ${bdr}`, borderRadius: 6, padding: 10, margin: "0 0 14px", textAlign: "left", whiteSpace: "pre-wrap", maxHeight: 140, overflow: "auto", fontFamily: "'JetBrains Mono',monospace" }}>{error.message || String(error)}</pre>
          <button type="button" onClick={() => this.setState({ error: null })} style={{ padding: "8px 18px", borderRadius: 7, border: "none", background: acc, color: "#fff", fontSize: 12, fontWeight: 600, cursor: "pointer" }}>Try again</button>
        </div>
      </div>
    );
  }
}
