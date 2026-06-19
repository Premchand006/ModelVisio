import preset from "@modelvisio/core/tailwind-preset.js";

/** @type {import('tailwindcss').Config} */
export default {
  presets: [preset],
  content: [
    "./index.html",
    "./src/**/*.{ts,tsx}",
    // Scan core so Tailwind keeps the classes used by the shared component library.
    "../../packages/core/src/**/*.{ts,tsx}",
  ],
};
