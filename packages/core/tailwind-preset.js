/**
 * Shared Tailwind preset for ModelVisio. Every app's tailwind.config
 * extends this so the dark/light theme and design tokens stay identical across
 * the web, desktop, and VS Code shells.
 */
export default {
  darkMode: "class",
  theme: {
    extend: {
      fontFamily: {
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
    },
  },
  plugins: [],
};
