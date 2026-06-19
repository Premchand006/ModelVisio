import type { ReactNode } from "react";
import { useT } from "../theme/ThemeContext";

export function Badge({ children, c }: { children: ReactNode; c?: string }) {
  const t = useT();
  const color = c || t.acc;
  return (
    <span
      style={{
        display: "inline-block", padding: "1px 6px", borderRadius: 3, fontSize: 9,
        fontWeight: 700, background: color + "18", color, border: `1px solid ${color}30`,
        fontFamily: "'JetBrains Mono',monospace",
      }}
    >
      {children}
    </span>
  );
}
