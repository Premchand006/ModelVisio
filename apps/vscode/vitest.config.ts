import { defineConfig } from "vitest/config";

// Pin the test root to this package. Covers both host-*.test.ts (pure
// extension-host logic, no `vscode` module) and webview-*.test.ts.
export default defineConfig({
  test: { environment: "node", include: ["test/**/*.test.ts"] },
});
