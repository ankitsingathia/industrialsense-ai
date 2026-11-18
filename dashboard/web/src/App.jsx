import { useEffect, useMemo, useRef, useState } from "react";
import Chart from "./Chart.jsx";
import { formatAt, formatHours } from "./time.js";

const SPEEDS = [
  { label: "1 h/s", step: 6 },
  { label: "6 h/s", step: 36 },
  { label: "1 day/s", step: 144 },
];

function useJson(url) {
  const [state, setState] = useState({ data: null, error: null });
  useEffect(() => {
    if (!url) return;
    let live = true;
    setState({ data: null, error: null });
    fetch(url)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`${r.status} ${r.statusText}`))))
      .then((data) => live && setState({ data, error: null }))
      .catch((error) => live && setState({ data: null, error }));
    return () => {
      live = false;
    };
  }, [url]);
  return state;
}

function StatusCard({ b, i, hours, selected, onSelect }) {
  const inAlarm = b.alarm[i] === 1;
  let since = null;
  if (inAlarm) {
    let s = i;
    while (s > 0 && b.alarm[s - 1] === 1) s--;
    since = hours[i] - hours[s];
  }
  const ratio = b.rms[i] / b.threshold[i];
  return (
    <button className={`card ${selected ? "selected" : ""}`} onClick={onSelect} aria-pressed={selected}>
      <div className="card-head">
        <span className="card-title">Bearing {b.bearing}</span>
        <span className={`status ${inAlarm ? "critical" : "good"}`}>
          <span aria-hidden="true">{inAlarm ? "▲" : "●"}</span> {inAlarm ? "Alarm" : "Normal"}
        </span>
      </div>
      <div className="card-value">
        {b.rms[i].toFixed(3)} <span className="unit">g RMS</span>
      </div>
      <div className="card-meta">
        {Math.round(ratio * 100)}% of threshold
        {since != null && <> · on for {formatHours(since)}</>}
      </div>
      <div className="card-outcome">
        Outcome at end of test: {b.failed ? `failed (${b.mode})` : "survived"}
      </div>
    </button>
  );
}

function AlarmLog({ data, cursorH, onPick }) {
  const rows = data.bearings
    .flatMap((b) => b.events.map((e) => ({ ...e, bearing: b.bearing })))
    .filter((e) => e.h <= cursorH)
    .sort((a, b) => b.h - a.h);
  if (!rows.length) return <p className="empty">No alarms yet in this replay.</p>;
  return (
    <table className="log">
      <thead>
        <tr>
          <th>When</th>
          <th>Bearing</th>
          <th>What moved most, vs this bearing's first 32 h</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((e) => (
          <tr key={`${e.bearing}-${e.h}`} onClick={() => onPick(e)} tabIndex={0} onKeyDown={(k) => k.key === "Enter" && onPick(e)}>
            <td className="num">{formatAt(data.start, e.h)}</td>
            <td>{e.bearing}</td>
            <td>
              {e.why.map((w, j) => (
                <span key={w.feature} className="why">
                  {w.label} <strong>{w.z > 0 ? "+" : "−"}{Math.abs(w.z).toFixed(1)}σ</strong>
                  {j < e.why.length - 1 ? ", " : ""}
                </span>
              ))}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function DataTable({ hours, b, start, upto }) {
  // one row every 6 hours of test time, up to the replay cursor
  const rows = [];
  let next = 0;
  for (let i = 0; i <= upto; i++) {
    if (hours[i] >= next) {
      rows.push(i);
      next = hours[i] + 6;
    }
  }
  return (
    <div className="table-scroll">
      <table className="log">
        <thead>
          <tr>
            <th>Time</th>
            <th className="num">RMS (g)</th>
            <th className="num">Threshold (g)</th>
            <th>Alarm</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((i) => (
            <tr key={i}>
              <td className="num">{formatAt(start, hours[i])}</td>
              <td className="num">{b.rms[i].toFixed(3)}</td>
              <td className="num">{b.threshold[i].toFixed(3)}</td>
              <td>{b.alarm[i] ? "yes" : "no"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function App() {
  const tests = useJson("/api/tests");
  const [test, setTest] = useState("test2");
  const { data, error } = useJson(`/api/tests/${test}`);
  const [cursor, setCursor] = useState(0);
  const [bearing, setBearing] = useState(1);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [showTable, setShowTable] = useState(false);
  const timer = useRef(null);

  useEffect(() => {
    if (!data) return;
    setCursor(data.hours.length - 1);
    setBearing(data.bearings.find((b) => b.failed)?.bearing ?? 1);
    setPlaying(false);
  }, [data]);

  useEffect(() => {
    if (!playing || !data) return undefined;
    timer.current = setInterval(() => {
      setCursor((c) => {
        const n = Math.min(c + SPEEDS[speed].step / 10, data.hours.length - 1);
        if (n >= data.hours.length - 1) setPlaying(false);
        return n;
      });
    }, 100);
    return () => clearInterval(timer.current);
  }, [playing, speed, data]);

  const i = Math.floor(cursor);
  const sel = useMemo(() => data?.bearings.find((b) => b.bearing === bearing), [data, bearing]);
  const info = tests.data?.find((t) => t.test === test);

  function play() {
    if (!data) return;
    if (i >= data.hours.length - 1) setCursor(0);
    setPlaying((p) => !p);
  }

  function pick(e) {
    setBearing(e.bearing);
    const k = data.hours.findIndex((h) => h >= e.h);
    setCursor(k < 0 ? data.hours.length - 1 : k);
    setPlaying(false);
  }

  return (
    <div className="page">
      <header className="top">
        <div>
          <h1>Bearing health</h1>
          <p className="sub">
            Replay of the NASA IMS run-to-failure tests through the stage 2b alarm. This is recorded test-rig data, not a live plant.
          </p>
        </div>
        <div className="segmented" role="tablist" aria-label="Test">
          {(tests.data ?? []).map((t) => (
            <button key={t.test} role="tab" aria-selected={t.test === test} className={t.test === test ? "on" : ""} onClick={() => setTest(t.test)}>
              {t.test.replace("test", "Test ")}
            </button>
          ))}
        </div>
      </header>

      {error && <p className="error">Could not load {test}: {error.message}. Is the API running on port 8765?</p>}
      {tests.error && <p className="error">Could not reach the API: {tests.error.message}</p>}

      {data && sel && (
        <>
          <section className="controls" aria-label="Replay">
            <button className="primary" onClick={play}>
              {playing ? "Pause" : i >= data.hours.length - 1 ? "Replay from start" : "Play"}
            </button>
            <div className="segmented small" aria-label="Speed">
              {SPEEDS.map((s, k) => (
                <button key={s.label} className={k === speed ? "on" : ""} onClick={() => setSpeed(k)} aria-pressed={k === speed}>
                  {s.label}
                </button>
              ))}
            </div>
            <input
              type="range"
              min={0}
              max={data.hours.length - 1}
              value={i}
              onChange={(e) => {
                setCursor(+e.target.value);
                setPlaying(false);
              }}
              aria-label="Replay position"
            />
            <div className="clock">
              <strong>{formatAt(data.start, data.hours[i])}</strong>
              <span>
                {formatHours(data.hours[i])} into the test{info ? ` of ${formatHours(info.hours)}` : ""}
              </span>
            </div>
          </section>

          <section className="cards">
            {data.bearings.map((b) => (
              <StatusCard key={b.bearing} b={b} i={i} hours={data.hours} selected={b.bearing === bearing} onSelect={() => setBearing(b.bearing)} />
            ))}
          </section>

          <section className="panel">
            <div className="panel-head">
              <h2>Bearing {sel.bearing}: vibration RMS against its alarm threshold</h2>
              <button className="link" onClick={() => setShowTable((v) => !v)}>
                {showTable ? "Show chart" : "Show as table"}
              </button>
            </div>
            <p className="note">
              The threshold is learned from this bearing's first {data.reference_hours} h and then follows slow drift.
              Shaded spans are alarms; gaps in the line are times the rig was stopped.{sel.latched_h != null && ` The alarm latched at ${formatHours(sel.latched_h)}.`}
            </p>
            {showTable ? (
              <DataTable hours={data.hours} b={sel} start={data.start} upto={i} />
            ) : (
              <Chart hours={data.hours} bearing={sel} cursor={i} start={data.start} refHours={data.reference_hours} />
            )}
            <div className="legend">
              <span><span className="swatch swatch-rms" /> RMS</span>
              <span><span className="swatch swatch-threshold" /> Alarm threshold</span>
              <span><span className="swatch swatch-alarm" /> Alarm</span>
              <span><span className="swatch swatch-ref" /> Learning period</span>
            </div>
          </section>

          <section className="panel">
            <h2>Alarm log</h2>
            <p className="note">
              For each alarm, the three features furthest from the bearing's own healthy reference, in standard deviations.
              Click a row to jump there.
            </p>
            <AlarmLog data={data} cursorH={data.hours[i]} onPick={pick} />
          </section>
        </>
      )}

      <footer className="foot">
        IndustrialSense AI, stage 5. Data: NASA IMS bearing dataset (University of Cincinnati). Alarm: RMS with a
        drift-following threshold, 72 h time constant, 3 snapshots in a row.
      </footer>
    </div>
  );
}
