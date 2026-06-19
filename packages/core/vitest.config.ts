import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

// React plugin transforms the .tsx components (and lets Vite resolve the PNG
// asset import in App.tsx); `node` env is enough since render smoke tests use
// react-dom/server (no DOM needed).
export default defineConfig({
  plugins: [react()],
  test: { environment: "node", include: ["test/**/*.test.{ts,tsx}"] },
});
