import { useEffect, useMemo, useRef, useState } from "react";
import { formatAt } from "./time.js";

const H = 300;
const M = { top: 16, right: 92, bottom: 34, left: 56 };
const GAP_H = 3; // break the line where the rig was stopped for more than 3 hours

function niceTicks(max, n = 4) {
  const raw = max / n;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw);
  const out = [];
  for (let v = 0; v <= max + 1e-9; v += step) out.push(+v.toFixed(6));
  return out;
}

function path(xs, ys, upto, sx, sy, hours) {
  let d = "";
  for (let i = 0; i <= upto; i++) {
    const move = i === 0 || hours[i] - hours[i - 1] > GAP_H;
    d += `${move ? "M" : "L"}${sx(xs[i]).toFixed(1)},${sy(ys[i]).toFixed(1)}`;
  }
  return d;
}

// Alarm spans as [startIndex, endIndex] pairs, up to the replay cursor.
function spans(alarm, upto) {
  const out = [];
  let s = -1;
  for (let i = 0; i <= upto; i++) {
    if (alarm[i] && s < 0) s = i;
    if ((!alarm[i] || i === upto) && s >= 0) {
      out.push([s, alarm[i] ? i : i - 1]);
      s = -1;
    }
  }
  return out;
}

function useWidth(ref) {
  const [w, setW] = useState(880);
  useEffect(() => {
    const ro = new ResizeObserver(([e]) => setW(Math.max(320, Math.round(e.contentRect.width))));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, [ref]);
  return w;
}

export default function Chart({ hours, bearing, cursor, start, refHours }) {
  const svgRef = useRef(null);
  const wrapRef = useRef(null);
  const W = useWidth(wrapRef);
  const [hover, setHover] = useState(null);
  const { rms, threshold, alarm } = bearing;

  const geo = useMemo(() => {
    const xMax = hours[hours.length - 1];
    const yMax = Math.max(...rms, ...threshold) * 1.08;
    const sx = (h) => M.left + (h / xMax) * (W - M.left - M.right);
    const sy = (v) => H - M.bottom - (v / yMax) * (H - M.top - M.bottom);
    const days = xMax / 24;
    const maxTicks = Math.max(3, Math.floor((W - M.left - M.right) / 70));
    const dayStep = [1, 2, 5, 7, 14].find((s) => days / s <= maxTicks) ?? 14;
    const xTicks = [];
    for (let d = 0; d * 24 <= xMax; d += dayStep) xTicks.push(d);
    return { xMax, yMax, sx, sy, xTicks, yTicks: niceTicks(yMax / 1.08) };
  }, [hours, rms, threshold, W]);

  const { sx, sy } = geo;
  const upto = Math.min(cursor, hours.length - 1);
  const last = upto;

  function onMove(e) {
    const r = svgRef.current.getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width) * W;
    const h = ((x - M.left) / (W - M.left - M.right)) * geo.xMax;
    // nearest snapshot that the replay has reached
    let lo = 0;
    let hi = upto;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (hours[mid] < h) lo = mid + 1;
      else hi = mid;
    }
    const i = lo > 0 && Math.abs(hours[lo - 1] - h) < Math.abs(hours[lo] - h) ? lo - 1 : lo;
    if (h < 0 || h > geo.xMax) setHover(null);
    else setHover(i);
  }

  const hx = hover != null ? sx(hours[hover]) : 0;
  const tipLeft = hover != null && hx > W * 0.62;

  return (
    <div className="chart-wrap" ref={wrapRef}>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        className="chart"
        role="img"
        aria-label={`Vibration RMS and alarm threshold for bearing ${bearing.bearing}`}
        onPointerMove={onMove}
        onPointerLeave={() => setHover(null)}
      >
        {/* reference period */}
        <rect
          x={sx(0)}
          y={M.top}
          width={sx(refHours) - sx(0)}
          height={H - M.top - M.bottom}
          className="ref-band"
        />
        <text x={sx(refHours / 2)} y={M.top + 12} className="ref-label" textAnchor="middle">
          learning
        </text>

        {spans(alarm, upto).map(([a, b]) => (
          <rect
            key={a}
            x={sx(hours[a])}
            y={M.top}
            width={Math.max(2, sx(hours[b]) - sx(hours[a]))}
            height={H - M.top - M.bottom}
            className="alarm-band"
          />
        ))}

        {geo.yTicks.map((v) => (
          <g key={v}>
            <line x1={M.left} x2={W - M.right} y1={sy(v)} y2={sy(v)} className="grid" />
            <text x={M.left - 8} y={sy(v) + 4} textAnchor="end" className="tick">
              {v}
            </text>
          </g>
        ))}
        {geo.xTicks.map((d) => (
          <text key={d} x={sx(d * 24)} y={H - M.bottom + 18} textAnchor="middle" className="tick">
            {d === 0 ? "start" : `day ${d}`}
          </text>
        ))}
        <line x1={M.left} x2={W - M.right} y1={sy(0)} y2={sy(0)} className="axis" />
        <text x={14} y={M.top + (H - M.top - M.bottom) / 2} className="axis-title" transform={`rotate(-90 14 ${M.top + (H - M.top - M.bottom) / 2})`} textAnchor="middle">
          RMS (g)
        </text>

        <path d={path(hours, threshold, upto, sx, sy, hours)} className="line-threshold" />
        <path d={path(hours, rms, upto, sx, sy, hours)} className="line-rms" />

        {/* direct labels at the replay cursor */}
        <text x={sx(hours[last]) + 8} y={sy(rms[last]) + 4} className="direct-label">
          RMS
        </text>
        <text x={sx(hours[last]) + 8} y={sy(threshold[last]) + 4} className="direct-label muted">
          threshold
        </text>

        <line x1={sx(hours[upto])} x2={sx(hours[upto])} y1={M.top} y2={H - M.bottom} className="cursor" />

        {hover != null && (
          <g pointerEvents="none">
            <line x1={hx} x2={hx} y1={M.top} y2={H - M.bottom} className="crosshair" />
            <circle cx={hx} cy={sy(rms[hover])} r={4} className="dot-rms" />
            <circle cx={hx} cy={sy(threshold[hover])} r={4} className="dot-threshold" />
          </g>
        )}
      </svg>

      {hover != null && (
        <div
          className="tooltip"
          style={{
            left: `${(hx / W) * 100}%`,
            transform: tipLeft ? "translateX(calc(-100% - 12px))" : "translateX(12px)",
          }}
        >
          <div className="tip-time">{formatAt(start, hours[hover])}</div>
          <div className="tip-row">
            <span className="swatch swatch-rms" />
            <strong>{rms[hover].toFixed(3)} g</strong> RMS
          </div>
          <div className="tip-row">
            <span className="swatch swatch-threshold" />
            <strong>{threshold[hover].toFixed(3)} g</strong> threshold
          </div>
          <div className="tip-row">{alarm[hover] ? "In alarm" : "No alarm"}</div>
        </div>
      )}
    </div>
  );
}
