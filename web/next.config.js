/** @type {import('next').NextConfig} */

/**
 * `output: "standalone"` is opt-in via NEXT_OUTPUT_STANDALONE rather than
 * always-on.
 *
 * infra/web.Dockerfile copies `.next/standalone` and needs it; Vercel builds
 * its own serverless output and does not. Turning it on unconditionally would
 * make the Docker image work but ship a second, unused copy of the traced
 * runtime on every Vercel build — so the Dockerfile sets the flag and Vercel
 * leaves it unset.
 */
const nextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
  ...(process.env.NEXT_OUTPUT_STANDALONE === "true"
    ? { output: "standalone" }
    : {}),
  async headers() {
    // A console that renders customer financial data should not be framable,
    // and should not leak referrers to third parties.
    return [
      {
        source: "/:path*",
        headers: [
          { key: "X-Frame-Options", value: "DENY" },
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
          {
            key: "Permissions-Policy",
            value: "camera=(), microphone=(), geolocation=()",
          },
          // HSTS is set here rather than at the edge so the guarantee travels
          // with the app regardless of where it is hosted.
          {
            key: "Strict-Transport-Security",
            value: "max-age=63072000; includeSubDomains; preload",
          },
        ],
      },
    ];
  },
};

module.exports = nextConfig;
