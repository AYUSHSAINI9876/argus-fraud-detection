import type { Config } from "tailwindcss";

/**
 * Design language: dark-first, dense, instrument-panel.
 *
 * A fraud console is read for hours at a time under time pressure, so the
 * palette is deliberately low-chroma with colour reserved almost entirely for
 * risk encoding. If everything is coloured, nothing reads as urgent.
 */
const config: Config = {
  darkMode: "class",
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Neutral surfaces — the panel itself recedes.
        surface: {
          0: "#0a0c10",
          1: "#11141a",
          2: "#171b23",
          3: "#1f242e",
          4: "#2a303c",
        },
        ink: {
          hi: "#e8ecf2",
          mid: "#9aa5b6",
          lo: "#5f6b7d",
        },
        // Risk ramp. Sequential, perceptually ordered, colour-blind safe
        // (avoids red/green as the sole distinction — decisions also carry
        // an icon and a text label).
        risk: {
          minimal: "#3b82f6",
          low: "#14b8a6",
          moderate: "#eab308",
          elevated: "#f97316",
          severe: "#ef4444",
        },
        accent: {
          DEFAULT: "#5b8def",
          muted: "#2d4a80",
        },
      },
      fontFamily: {
        sans: ["var(--font-sans)", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "ui-monospace", "SFMono-Regular", "monospace"],
      },
      fontSize: {
        "2xs": ["0.6875rem", { lineHeight: "1rem" }],
      },
      borderRadius: {
        panel: "0.625rem",
      },
      keyframes: {
        "pulse-risk": {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.55" },
        },
        "slide-up": {
          from: { opacity: "0", transform: "translateY(4px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
      },
      animation: {
        "pulse-risk": "pulse-risk 2s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "slide-up": "slide-up 160ms ease-out",
      },
    },
  },
  plugins: [],
};

export default config;
