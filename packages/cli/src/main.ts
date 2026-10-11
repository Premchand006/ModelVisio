// Process entry point for the `modelvisio` binary.
import { run } from "./cli";

run(process.argv.slice(2), {
  stdout: (s) => process.stdout.write(s),
  stderr: (s) => process.stderr.write(s),
  cwd: process.cwd(),
  env: process.env,
}).then(
  (code) => { process.exitCode = code; },
  (e: unknown) => { process.stderr.write(`modelvisio: ${String(e)}\n`); process.exitCode = 2; },
);
