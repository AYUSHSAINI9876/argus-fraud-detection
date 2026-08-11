import { FlatCompat } from "@eslint/eslintrc";

/**
 * ESLint flat config.
 *
 * `next lint` is deprecated in Next 15.5 and removed in 16. With no config
 * present it drops into an interactive setup prompt, which in CI means the
 * job hangs and then fails with exit 1 — the failure looks like a lint error
 * but is actually a missing config.
 *
 * Using the ESLint CLI against a flat config is the forward-compatible
 * answer: it survives the Next 16 removal and runs non-interactively.
 *
 * FlatCompat is required because `eslint-config-next` still ships in the
 * legacy eslintrc format.
 */
const compat = new FlatCompat({
  baseDirectory: import.meta.dirname,
});

const config = [
  {
    ignores: [
      ".next/**",
      "node_modules/**",
      "out/**",
      "next-env.d.ts",
    ],
  },
  ...compat.extends("next/core-web-vitals", "next/typescript"),
  {
    rules: {
      // The API client and chart tooltips receive genuinely untyped payloads
      // from Recharts and from JSON responses whose shape is asserted at the
      // boundary instead. Warn rather than error so real issues stay visible.
      "@typescript-eslint/no-explicit-any": "warn",
      "@typescript-eslint/no-unused-vars": [
        "error",
        { argsIgnorePattern: "^_", varsIgnorePattern: "^_" },
      ],
    },
  },
];

export default config;
