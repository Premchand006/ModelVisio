import { defineConfig } from "vitest/config";

// Pin the test root to this package (see packages/parsers/vitest.config.ts).
export default defineConfig({
  test: { environment: "node", include: ["test/**/*.test.ts"], testTimeout: 60_000 },
});
