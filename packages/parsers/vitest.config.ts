import { defineConfig } from "vitest/config";

// Pin the test root to this package so vitest never picks up a config from a
// parent directory (e.g. when the repo is checked out inside another project).
export default defineConfig({
  test: { environment: "node", include: ["test/**/*.test.ts"] },
});
