FROM node:22-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json* ./

# Strict ci — matching the CI job. A silent fallback to `npm install` would
# resolve a different tree than the committed lock file, which is the drift a
# lock file exists to prevent.
#
# The retry settings are not superstition: this install pulls ~1200 packages,
# and a single ECONNRESET part-way through fails the whole layer. npm's
# defaults give up quickly, which turns a transient blip into a failed build.
RUN npm config set fetch-retries 5 \
    && npm config set fetch-retry-maxtimeout 120000 \
    && npm config set fetch-timeout 600000 \
    && npm ci

FROM node:22-alpine AS builder
WORKDIR /app
ARG NEXT_PUBLIC_API_URL
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL
ENV NEXT_TELEMETRY_DISABLED=1
# Opts the build into `output: "standalone"` (see next.config.js). Without
# this the runner stage below copies a directory that was never emitted.
ENV NEXT_OUTPUT_STANDALONE=true
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

FROM node:22-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production NEXT_TELEMETRY_DISABLED=1 PORT=3000 HOSTNAME=0.0.0.0

RUN addgroup -g 10001 -S nodejs && adduser -S nextjs -u 10001

# Standalone output ships only the traced dependencies — a fraction of the
# node_modules tree, and a much smaller attack surface.
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static
COPY --from=builder --chown=nextjs:nodejs /app/public ./public

USER nextjs
EXPOSE 3000
CMD ["node", "server.js"]
