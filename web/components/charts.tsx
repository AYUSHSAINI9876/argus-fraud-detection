"use client";

import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { format, parseISO } from "date-fns";
import type { TimeseriesPoint } from "@/lib/api";

/**
 * Chart styling notes:
 *
 * Grid lines are drawn at very low contrast and only on the Y axis. In a
 * dense dark console, a full grid competes with the data for attention and
 * the series stops reading as the primary object.
 *
 * Blocks and reviews are stacked because they are parts of one whole (total
 * intervention volume). Total scored is a separate line, not a third stack
 * layer — it is a different magnitude entirely and stacking it would flatten
 * the two series that matter.
 */

const AXIS = {
  stroke: "#5f6b7d",
  fontSize: 11,
  tickLine: false,
  axisLine: false,
};

function ChartTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="panel px-3 py-2 shadow-lg text-xs">
      <div className="text-ink-lo mb-1.5 tnum">
        {format(parseISO(label), "d MMM, HH:mm")}
      </div>
      {payload.map((p: any) => (
        <div key={p.dataKey} className="flex items-center justify-between gap-4">
          <span className="flex items-center gap-1.5 text-ink-mid capitalize">
            <span
              className="w-2 h-2 rounded-full"
              style={{ background: p.color }}
            />
            {p.name}
          </span>
          <span className="tnum font-medium text-ink-hi">
            {typeof p.value === "number" ? p.value.toLocaleString() : p.value}
          </span>
        </div>
      ))}
    </div>
  );
}

export function DecisionVolumeChart({ points }: { points: TimeseriesPoint[] }) {
  if (!points.length) {
    return (
      <div className="h-64 flex items-center justify-center text-sm text-ink-lo">
        No data in this window
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={256}>
      <AreaChart data={points} margin={{ top: 4, right: 4, bottom: 0, left: -12 }}>
        <defs>
          <linearGradient id="gReview" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#eab308" stopOpacity={0.35} />
            <stop offset="100%" stopColor="#eab308" stopOpacity={0.03} />
          </linearGradient>
          <linearGradient id="gBlock" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#ef4444" stopOpacity={0.35} />
            <stop offset="100%" stopColor="#ef4444" stopOpacity={0.03} />
          </linearGradient>
        </defs>

        <CartesianGrid stroke="#1f242e" vertical={false} />
        <XAxis
          dataKey="t"
          {...AXIS}
          tickFormatter={(v) => format(parseISO(v), "HH:mm")}
          minTickGap={40}
        />
        <YAxis {...AXIS} width={48} />
        <Tooltip content={<ChartTooltip />} cursor={{ stroke: "#2a303c" }} />

        <Area
          type="monotone"
          dataKey="reviews"
          name="reviews"
          stackId="1"
          stroke="#eab308"
          strokeWidth={1.5}
          fill="url(#gReview)"
        />
        <Area
          type="monotone"
          dataKey="blocks"
          name="blocks"
          stackId="1"
          stroke="#ef4444"
          strokeWidth={1.5}
          fill="url(#gBlock)"
        />
        <Line
          type="monotone"
          dataKey="count"
          name="scored"
          stroke="#5b8def"
          strokeWidth={1.5}
          dot={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}

/**
 * Precision-recall curve. Plotted with recall on X and precision on Y, with
 * the axis fixed to [0,1] on both — auto-scaling a PR curve makes a weak
 * model look strong, which defeats the purpose of showing it.
 */
export function PRCurveChart({
  curves,
}: {
  curves: Record<string, { precision: number; recall: number }[]>;
}) {
  const colours: Record<string, string> = {
    xgboost_risk: "#5b8def",
    logistic_baseline: "#9aa5b6",
  };

  const merged = (curves[Object.keys(curves)[0]] ?? []).map((_, i) => {
    const row: Record<string, number> = {};
    for (const [name, pts] of Object.entries(curves)) {
      if (pts[i]) {
        row.recall = pts[i].recall;
        row[name] = pts[i].precision;
      }
    }
    return row;
  });

  return (
    <ResponsiveContainer width="100%" height={240}>
      <AreaChart data={merged} margin={{ top: 4, right: 8, bottom: 0, left: -12 }}>
        <CartesianGrid stroke="#1f242e" />
        <XAxis
          dataKey="recall"
          {...AXIS}
          domain={[0, 1]}
          type="number"
          tickFormatter={(v) => v.toFixed(1)}
          label={{ value: "Recall", position: "insideBottom", offset: -2, fill: "#5f6b7d", fontSize: 11 }}
        />
        <YAxis {...AXIS} domain={[0, 1]} width={48} tickFormatter={(v) => v.toFixed(1)} />
        <Tooltip content={<ChartTooltip />} />
        {Object.keys(curves).map((name) => (
          <Line
            key={name}
            type="monotone"
            dataKey={name}
            stroke={colours[name] ?? "#14b8a6"}
            strokeWidth={1.75}
            dot={false}
          />
        ))}
      </AreaChart>
    </ResponsiveContainer>
  );
}
