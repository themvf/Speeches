"use client";

import { AreaChart, Area, ResponsiveContainer, Tooltip } from "recharts";
import type { TrendSparklinePoint } from "@/lib/server/types";

interface SparklineChartProps {
  data: TrendSparklinePoint[];
  color?: string;
}

export function SparklineChart({ data, color = "var(--accent)" }: SparklineChartProps) {
  if (!data || data.length === 0) {
    return <div className="h-9 w-[120px] rounded opacity-30 bg-[color:var(--line)]" />;
  }

  return (
    <div className="h-9 w-[120px] shrink-0">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 2, right: 0, left: 0, bottom: 2 }}>
          <defs>
            <linearGradient id="sparkGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={color} stopOpacity={0.35} />
              <stop offset="95%" stopColor={color} stopOpacity={0} />
            </linearGradient>
          </defs>
          <Tooltip
            content={({ active, payload, label }) => {
              if (!active || !payload?.length) return null;
              return (
                <div className="rounded border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.95)] px-2 py-1 text-xs text-[color:var(--ink)]">
                  <p className="text-[color:var(--ink-faint)]">{label}</p>
                  <p className="font-semibold">{payload[0].value} mentions</p>
                </div>
              );
            }}
          />
          <Area
            type="monotone"
            dataKey="count"
            stroke={color}
            strokeWidth={1.5}
            fill="url(#sparkGrad)"
            dot={false}
            isAnimationActive={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
