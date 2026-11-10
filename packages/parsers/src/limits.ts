// Size limits the UI needs on the main thread. Its own module (and the
// "./limits" subpath) so importing it does not pull in the parsers.

/** Files at or above this size are parsed progressively by the UI. */
export const STREAM_THRESHOLD_BYTES = 100 * 1024 * 1024;
