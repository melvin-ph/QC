import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Atom,
  Play,
  RotateCcw,
  Zap,
  Cpu,
  Scale,
  CheckCircle2,
  ChevronRight,
  FlaskConical,
  CircleHelp,
  Activity,
  LineChart,
  BarChart3,
  TrendingDown
} from "lucide-react";
import "./style.css";

const DEFAULT_WEIGHTS = [3, 1, 1, 2, 2, 1];

// Statevector convention: index is the computational-basis bitstring interpreted as an integer.
// H_P is diagonal, so its evolution is applied as a phase. H_0=-sum(X_i), whose terms commute,
// so its evolution is applied as independent single-qubit X rotations. A first-order product
// formula approximates exp(-i[(1-s)H0+sHP]dt) at each midpoint.
async function simulateAdiabatic(weights, totalTime = 12, steps = 360, onFrame = () => {}, shouldStop = () => false) {
  const n = weights.length, size = 1 << n;
  if (n > 10) throw new Error("Use 10 workload items or fewer for this browser simulation.");
  let state = new Float64Array(size * 2);
  const amp = 1 / Math.sqrt(size);
  for (let i = 0; i < size; i++) state[2 * i] = amp;

  const energies = new Float64Array(size);
  const bitStrings = [];
  for (let idx = 0; idx < size; idx++) {
    const bits = idx.toString(2).padStart(n, "0");
    bitStrings.push(bits);
    let d = 0;
    for (let q = 0; q < n; q++) d += weights[q] * (bits[q] === "0" ? 1 : -1);
    energies[idx] = d * d;
  }

  const dt = totalTime / steps;
  const frameEvery = Math.max(1, Math.floor(steps / 100));
  const normAndStats = (s, step) => {
    let norm = 0, energy = 0;
    const probs = new Float64Array(size);
    for (let i = 0; i < size; i++) {
      const re = state[2 * i], im = state[2 * i + 1];
      const p = re * re + im * im;
      probs[i] = p; norm += p; energy += p * energies[i];
    }
    if (norm > 0) {
      for (let i = 0; i < size; i++) probs[i] /= norm;
      energy /= norm;
    }
    const ranked = Array.from({ length: size }, (_, i) => i)
      .sort((a, b) => probs[b] - probs[a]).slice(0, 8)
      .map(i => ({ bits: bitStrings[i], probability: probs[i], energy: energies[i] }));
    const minEnergy = Math.min(...energies);
    let optimalProbability = 0;
    for (let i = 0; i < size; i++) if (energies[i] === minEnergy) optimalProbability += probs[i];
    onFrame({
      progress: Math.round(s * 100), s, step, steps, energyExpectation: energy,
      optimalProbability, topStates: ranked, probabilities: Array.from(probs),
      bitStrings, energies: Array.from(energies), minEnergy
    });
  };

  normAndStats(0, 0);
  for (let step = 0; step < steps; step++) {
    if (shouldStop()) return { cancelled: true };
    const s = (step + 0.5) / steps;

    // Apply exp(-i * s * HP * dt): HP is diagonal.
    for (let idx = 0; idx < size; idx++) {
      const theta = s * energies[idx] * dt;
      const c = Math.cos(theta), sn = Math.sin(theta);
      const re = state[2 * idx], im = state[2 * idx + 1];
      state[2 * idx] = re * c + im * sn;
      state[2 * idx + 1] = im * c - re * sn;
    }

    // Apply exp(-i * (1-s) * H0 * dt), H0=-sum X.
    const alpha = (1 - s) * dt;
    const c = Math.cos(alpha), sn = Math.sin(alpha);
    for (let q = 0; q < n; q++) {
      const mask = 1 << (n - 1 - q);
      for (let idx = 0; idx < size; idx++) {
        if (idx & mask) continue;
        const j = idx | mask;
        const ar = state[2 * idx], ai = state[2 * idx + 1];
        const br = state[2 * j], bi = state[2 * j + 1];
        state[2 * idx] = c * ar - sn * bi;
        state[2 * idx + 1] = c * ai + sn * br;
        state[2 * j] = c * br - sn * ai;
        state[2 * j + 1] = c * bi + sn * ar;
      }
    }
    if ((step + 1) % frameEvery === 0 || step === steps - 1) {
      normAndStats((step + 1) / steps, step + 1);
      await new Promise(resolve => setTimeout(resolve, 0));
    }
  }
  return { cancelled: false };
}

function bruteForce(weights) {
  const n = weights.length, total = weights.reduce((a, b) => a + b, 0);
  const rows = [];
  for (let mask = 0; mask < (1 << n); mask++) {
    let a = 0; const bits = mask.toString(2).padStart(n, "0");
    for (let i = 0; i < n; i++) if (bits[i] === "1") a += weights[i];
    rows.push({ bits, a, b: total - a, diff: Math.abs(a - (total - a)) });
  }
  const best = Math.min(...rows.map(r => r.diff));
  return { rows, best, optimal: rows.filter(r => r.diff === best) };
}

function EnergyGraph({ history, minEnergy = 0 }) {
  const width = 540;
  const height = 210;
  const padLeft = 45;
  const padRight = 45;
  const padTop = 20;
  const padBottom = 30;
  const plotW = width - padLeft - padRight;
  const plotH = height - padTop - padBottom;

  if (!history || history.length === 0) {
    return (
      <div className="svg-chart-wrap">
        <svg viewBox={`0 0 ${width} ${height}`} className="energy-svg" preserveAspectRatio="xMidYMid meet">
          {[0, 0.25, 0.5, 0.75, 1].map((f, i) => {
            const y = padTop + plotH * (1 - f);
            return (
              <g key={i}>
                <line x1={padLeft} y1={y} x2={padLeft + plotW} y2={y} stroke="#252a3d" strokeDasharray="3 3" />
                <text x={padLeft - 8} y={y + 3.5} textAnchor="end" fill="#58617d" fontSize="9" fontFamily="'DM Mono', monospace">
                  {(20 * f).toFixed(0)}
                </text>
                <text x={padLeft + plotW + 8} y={y + 3.5} textAnchor="start" fill="#58617d" fontSize="9" fontFamily="'DM Mono', monospace">
                  {`${Math.round(f * 100)}%`}
                </text>
              </g>
            );
          })}
          {[0, 0.25, 0.5, 0.75, 1].map((f, i) => {
            const x = padLeft + plotW * f;
            return (
              <g key={i}>
                <line x1={x} y1={padTop} x2={x} y2={padTop + plotH} stroke="#202436" strokeDasharray="2 2" />
                <text x={x} y={padTop + plotH + 18} textAnchor="middle" fill="#58617d" fontSize="9" fontFamily="'DM Mono', monospace">
                  {`s=${f.toFixed(2)}`}
                </text>
              </g>
            );
          })}
          <text x={width / 2} y={padTop + plotH / 2 - 5} textAnchor="middle" fill="#7882a0" fontSize="11" fontFamily="'DM Mono', monospace">
            Awaiting simulation run
          </text>
          <text x={width / 2} y={padTop + plotH / 2 + 14} textAnchor="middle" fill="#535c77" fontSize="9" fontFamily="'DM Mono', monospace">
            Press &quot;Start simulation&quot; to plot real-time ⟨HP⟩ energy decay
          </text>
        </svg>
      </div>
    );
  }

  const maxE = Math.max(1, ...history.map(h => h.energyExpectation));
  const yMax = maxE <= 4 ? 4 : maxE <= 10 ? 10 : maxE <= 20 ? 20 : Math.ceil(maxE / 5) * 5;
  const yMin = 0;
  const yRange = yMax - yMin || 1;

  const getX = prog => padLeft + (Math.max(0, Math.min(100, prog)) / 100) * plotW;
  const getY = e => padTop + plotH - ((Math.max(0, e) - yMin) / yRange) * plotH;
  const getProbY = p => padTop + plotH - Math.max(0, Math.min(1, p)) * plotH;

  const energyPts = history.map(h => ({
    x: getX(h.progress),
    y: getY(h.energyExpectation),
    energy: h.energyExpectation,
    progress: h.progress
  }));

  const probPts = history.map(h => ({
    x: getX(h.progress),
    y: getProbY(h.optimalProbability)
  }));

  const lineD = "M " + energyPts.map(p => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" L ");
  const areaD = `${lineD} L ${energyPts[energyPts.length - 1].x.toFixed(1)},${(padTop + plotH).toFixed(1)} L ${energyPts[0].x.toFixed(1)},${(padTop + plotH).toFixed(1)} Z`;
  const probLineD = "M " + probPts.map(p => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" L ");

  const latestEnergy = energyPts[energyPts.length - 1];
  const latestProb = probPts[probPts.length - 1];

  return (
    <div className="svg-chart-wrap">
      <svg viewBox={`0 0 ${width} ${height}`} className="energy-svg" preserveAspectRatio="xMidYMid meet">
        <defs>
          <linearGradient id="energyStrokeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#9c9bff" />
            <stop offset="60%" stopColor="#a49aff" />
            <stop offset="100%" stopColor="#69e0b5" />
          </linearGradient>
          <linearGradient id="energyAreaGrad" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#9c9bff" stopOpacity="0.28" />
            <stop offset="100%" stopColor="#69e0b5" stopOpacity="0.01" />
          </linearGradient>
        </defs>

        {/* Horizontal grid lines & Y labels */}
        {[0, 0.25, 0.5, 0.75, 1].map((f, i) => {
          const y = padTop + plotH * (1 - f);
          const eVal = yMin + yRange * f;
          return (
            <g key={i}>
              <line x1={padLeft} y1={y} x2={padLeft + plotW} y2={y} stroke="#252a3d" strokeDasharray="3 3" />
              <text x={padLeft - 8} y={y + 3.5} textAnchor="end" fill="#717a96" fontSize="9" fontFamily="'DM Mono', monospace">
                {eVal >= 10 ? eVal.toFixed(0) : eVal.toFixed(1)}
              </text>
              <text x={padLeft + plotW + 8} y={y + 3.5} textAnchor="start" fill="#5f8c7b" fontSize="9" fontFamily="'DM Mono', monospace">
                {`${Math.round(f * 100)}%`}
              </text>
            </g>
          );
        })}

        {/* Vertical grid lines & X labels */}
        {[0, 0.25, 0.5, 0.75, 1].map((f, i) => {
          const x = padLeft + plotW * f;
          return (
            <g key={i}>
              <line x1={x} y1={padTop} x2={x} y2={padTop + plotH} stroke="#202436" strokeDasharray="2 2" />
              <text x={x} y={padTop + plotH + 18} textAnchor="middle" fill="#717a96" fontSize="9" fontFamily="'DM Mono', monospace">
                {f === 0 ? "s=0" : f === 1 ? "s=1" : `s=${f.toFixed(2)}`}
              </text>
            </g>
          );
        })}

        {/* Area fill under Energy Expectation */}
        <path d={areaD} fill="url(#energyAreaGrad)" />

        {/* Energy Expectation Line */}
        <path d={lineD} fill="none" stroke="url(#energyStrokeGrad)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />

        {/* Optimal probability line (dashed green) */}
        <path d={probLineD} fill="none" stroke="#69e0b5" strokeWidth="1.8" strokeDasharray="4 3" strokeOpacity="0.85" strokeLinecap="round" />

        {/* Tracker dot on Optimal Probability */}
        {latestProb && (
          <circle cx={latestProb.x} cy={latestProb.y} r="3.5" fill="#69e0b5" stroke="#121521" strokeWidth="1.5" />
        )}

        {/* Tracker dot on Energy Expectation */}
        {latestEnergy && (
          <g>
            <circle cx={latestEnergy.x} cy={latestEnergy.y} r="8" fill="none" stroke="#a49aff" strokeWidth="1" strokeOpacity="0.6" className="pulse-circle" />
            <circle cx={latestEnergy.x} cy={latestEnergy.y} r="4.5" fill="#a49aff" stroke="#121521" strokeWidth="2" />
            <text
              x={Math.max(padLeft + 45, Math.min(latestEnergy.x, padLeft + plotW - 40))}
              y={Math.max(padTop + 14, latestEnergy.y - 10)}
              textAnchor="middle"
              fill="#dcdbff"
              fontSize="10"
              fontWeight="600"
              fontFamily="'DM Mono', monospace"
            >
              {latestEnergy.energy.toFixed(2)}
            </text>
          </g>
        )}
      </svg>
    </div>
  );
}

function ProbabilityChart({ topStates, minEnergy = 0, frame }) {
  const displayStates = (!frame || !topStates || topStates.length === 0)
    ? Array.from({ length: 8 }, (_, i) => ({ bits: "—", probability: 0, energy: null }))
    : topStates.slice(0, 8);

  return (
    <div className="prob-card-list">
      {displayStates.map((s, i) => {
        const isOptimal = frame && s.energy === minEnergy;
        const diff = s.energy !== null ? Math.round(Math.sqrt(s.energy)) : null;
        const pct = (s.probability * 100).toFixed(2);

        return (
          <div className={`prob-card-row ${isOptimal ? "optimal" : ""}`} key={`${s.bits}-${i}`}>
            <div className="prob-card-label">
              <code>{s.bits}</code>
              {frame ? (
                isOptimal ? (
                  <span className="state-tag optimal">OPTIMAL (Δ=0)</span>
                ) : (
                  <span className="state-tag">Δ={diff}</span>
                )
              ) : (
                <span className="state-tag">—</span>
              )}
            </div>
            <div className="bar-track">
              <div
                className={`bar-fill ${isOptimal ? "optimal" : ""}`}
                style={{ width: `${Math.max(0, Math.min(100, s.probability * 100))}%` }}
              />
            </div>
            <div className="prob-card-val">
              <strong className={isOptimal ? "green" : ""}>{frame ? `${pct}%` : "0.00%"}</strong>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function App() {
  const [weights, setWeights] = useState(DEFAULT_WEIGHTS);
  const [inputText, setInputText] = useState(DEFAULT_WEIGHTS.join(", "));
  const [evolutionTime, setEvolutionTime] = useState(12);
  const [steps, setSteps] = useState(360);
  const [speed, setSpeed] = useState(2);
  const [status, setStatus] = useState("idle");
  const [frame, setFrame] = useState(null);
  const [history, setHistory] = useState([]);
  const [error, setError] = useState("");
  const [tab, setTab] = useState("overview");
  const [showAll, setShowAll] = useState(false);
  const cancelRef = useRef(false);
  const latestRef = useRef(null);
  const total = weights.reduce((a, b) => a + b, 0);
  const verification = useMemo(() => bruteForce(weights), [weights]);
  const n = weights.length;

  useEffect(() => { latestRef.current = frame; }, [frame]);

  const parseWeights = () => {
    const vals = inputText.split(",").map(s => s.trim()).filter(Boolean).map(Number);
    if (!vals.length || vals.some(v => !Number.isInteger(v) || v <= 0)) {
      setError("Enter positive whole numbers separated by commas."); return;
    }
    if (vals.length > 10) { setError("Use 10 workload items or fewer."); return; }
    setError(""); cancelRef.current = true;
    setWeights(vals); setStatus("idle"); setFrame(null); setHistory([]);
  };

  const run = async () => {
    setError(""); cancelRef.current = false; setStatus("running"); setFrame(null); setHistory([]);
    const localWeights = [...weights];
    const localSteps = steps;
    let lastPaint = 0;
    const historyBuffer = [];
    try {
      await new Promise(resolve => setTimeout(resolve, 30));
      const result = await simulateAdiabatic(localWeights, evolutionTime, localSteps, (f) => {
        if (cancelRef.current) return;
        historyBuffer.push({
          progress: f.progress,
          s: f.s,
          step: f.step,
          energyExpectation: f.energyExpectation,
          optimalProbability: f.optimalProbability
        });
        const now = performance.now();
        if (f.progress === 100 || now - lastPaint > (110 / speed)) {
          setFrame(f);
          setHistory([...historyBuffer]);
          lastPaint = now;
        }
      }, () => cancelRef.current);
      if (result.cancelled || cancelRef.current) { setStatus("paused"); return; }
      setHistory([...historyBuffer]);
      setStatus("complete");
    } catch (e) { setError(e.message || "Simulation failed."); setStatus("idle"); }
  };

  const reset = () => {
    cancelRef.current = true; setStatus("idle"); setFrame(null); setHistory([]); setError("");
  };

  const topStates = frame?.topStates || [];
  const optimalProbability = frame?.optimalProbability ?? 0;
  const finalState = status === "complete" && frame;
  const chosen = finalState ? [...topStates].sort((a,b) => {
    const aRow = verification.rows.find(r => r.bits === a.bits);
    const bRow = verification.rows.find(r => r.bits === b.bits);
    return (aRow?.diff ?? 99) - (bRow?.diff ?? 99) || b.probability-a.probability;
  })[0] : null;
  const allocation = chosen ? verification.rows.find(r => r.bits === chosen.bits) : null;

  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand"><div className="brand-icon"><Atom size={23}/></div><div><strong>Q-Balance</strong><span>QUANTUM LAB</span></div></div>
      <div className="nav-label">WORKSPACE</div>
      {[
        ["overview", Activity, "Overview"],
        ["simulator", Atom, "Simulator"],
        ["hamiltonian", FlaskConical, "Ising Hamiltonian"],
        ["verification", Scale, "Verification"]
      ].map(([id, Icon, label]) => <button key={id} className={`nav-item ${tab===id?"active":""}`} onClick={() => setTab(id)}><Icon size={17}/>{label}{tab===id&&<ChevronRight size={15} className="nav-arrow"/>}</button>)}
      <div className="sidebar-bottom"><div className="status-dot"/><span>LOCAL SIMULATION</span><p>No backend · Runs in browser</p></div>
    </aside>

    <main className="main">
      <header className="topbar"><div><div className="eyebrow">QUANTUM COMPUTING / NUMBER PARTITIONING</div><h1>{tab==="overview"?"Workload balancing":tab==="simulator"?"Adiabatic simulator":tab==="hamiltonian"?"Ising Hamiltonian":"Classical verification"}</h1><p>Explore how adiabatic evolution searches for balanced server assignments.</p></div><div className={`status-pill ${status}`}><span/>{status==="complete"?"SIMULATION COMPLETE":status==="running"?"EVOLVING":status==="paused"?"STOPPED":"READY"}</div></header>

      <section className="stats-grid">
        <Stat icon={<Cpu size={17}/>} label="QUBITS" value={n} note={`${n} workload items`}/>
        <Stat icon={<Zap size={17}/>} label="SEARCH SPACE" value={2**n} note="Possible assignments"/>
        <Stat icon={<Scale size={17}/>} label="BEST IMBALANCE" value={verification.best} note="Exact brute-force result" accent/>
        <Stat icon={<CheckCircle2 size={17}/>} label="OPTIMAL PARTITIONS" value={verification.optimal.length} note="Bitstrings incl. symmetry"/>
      </section>

      {tab==="overview" && <div className="content-grid">
        <section className="panel input-panel">
          <PanelTitle number="01" title="Workload configuration" subtitle="Set the task sizes assigned across two servers."/>
          <label className="field-label" htmlFor="weights">Workload values</label>
          <input id="weights" value={inputText} onChange={e=>setInputText(e.target.value)} onKeyDown={e=>e.key==="Enter"&&parseWeights()} placeholder="3, 1, 1, 2, 2, 1"/>
          <div className="workload-chips">{weights.map((w,i)=><div className="work-chip" key={`${i}-${w}`}><span>Q{i+1}</span><strong>{w}</strong></div>)}</div>
          <div className="field-help">Each item maps to one qubit. Bit 1 → Server A; bit 0 → Server B.</div>
          <button className="secondary-btn full" onClick={parseWeights}>Apply workloads</button>
          {error&&<div className="error-message">{error}</div>}
          <div className="settings-row"><label>Evolution time <b>{evolutionTime}</b></label><input type="range" min="2" max="30" value={evolutionTime} onChange={e=>setEvolutionTime(+e.target.value)}/></div>
          <div className="settings-row"><label>Time steps <b>{steps}</b></label><input type="range" min="120" max="800" step="40" value={steps} onChange={e=>setSteps(+e.target.value)}/></div>
          <div className="settings-row"><label>Animation speed</label><select value={speed} onChange={e=>setSpeed(+e.target.value)}><option value="1">Slow</option><option value="2">Normal</option><option value="4">Fast</option></select></div>
          <div className="action-row">
            <button className="primary-btn" disabled={status==="running"} onClick={run}><Play size={16} fill="currentColor"/>{status==="complete"?"Run again":"Start simulation"}</button>
            <button className="icon-btn" title="Stop and reset simulation" onClick={reset}><RotateCcw size={16}/></button>
          </div>
          <p className="fine-print"><CircleHelp size={13}/> Numerical simulation using a first-order product formula; no quantum hardware required.</p>
        </section>

        <section className="panel evolution-panel">
          <PanelTitle number="02" title="Adiabatic evolution" subtitle="The system transitions from an easy initial Hamiltonian to the workload objective."/>
          <div className="evolution-meta"><span>EVOLUTION PARAMETER <b>s</b></span><strong>{(frame?.s ?? 0).toFixed(3)}</strong></div>
          <div className="progress-track"><div className="progress-fill" style={{width:`${frame?.progress||0}%`}}/></div>
          <div className="progress-labels"><span>Initial state · s = 0</span><span>Problem state · s = 1</span></div>
          <div className="qubit-stage">
            <div className="stage-grid-glow"/>
            <div className="qubits">{weights.map((w,i)=>{
              const p = frame?.probabilities || [];
              const oneProb = p.reduce((sum,prob,idx)=>sum + (frame?.bitStrings[idx]?.[i]==="1"?prob:0),0);
              const angle = (oneProb*180).toFixed(0);
              return <div className="qubit" key={i}>
                <div className={`qubit-orb ${frame ? "measured":""}`} style={{"--tilt":`${angle}deg`}}>
                  <div className="orb-core">{frame ? (oneProb>.5?"1":"0"):"+"}</div><div className="orb-ring"/>
                </div>
                <span>q{i+1}</span><small>{frame?`${Math.round(oneProb*100)}% |1⟩`:"superposition"}</small>
              </div>;
            })}</div>
            <div className="stage-caption"><span className="pulse-dot"/>{status==="running"?"Evolving statevector":status==="complete"?"Evolution finished":status==="paused"?"Simulation stopped":"Initial superposition prepared"}</div>
          </div>
          <div className="energy-row"><div><span>ENERGY EXPECTATION</span><strong>{frame?frame.energyExpectation.toFixed(3):"—"}</strong></div><div><span>OPTIMAL-STATE PROBABILITY</span><strong>{frame?`${(optimalProbability*100).toFixed(2)}%`:"—"}</strong></div></div>
          <div className="chart-title"><span>Measurement distribution</span><small>{frame?`${frame.probabilities.length} basis states`:"Run simulation to calculate"}</small></div>
          <div className="prob-chart">{(frame?.topStates || Array.from({length:8},(_,i)=>({bits:"—",probability:0}))).slice(0,8).map((s,i)=>{
            const isOpt = frame && s.energy === frame.minEnergy;
            return (
              <div className={`prob-row ${isOpt ? "optimal" : ""}`} key={s.bits+i}>
                <code>{s.bits}</code>
                <div className="bar-track"><div className={`bar-fill ${isOpt ? "optimal" : ""}`} style={{width:`${Math.max(0,s.probability*100)}%`}}/></div>
                <span className={isOpt ? "green" : ""}>{(s.probability*100).toFixed(1)}%</span>
              </div>
            );
          })}</div>
        </section>
      </div>}

      {tab==="overview" && <section className="panel analytics-panel">
        <PanelTitle number="03" title="Evolution analytics" subtitle="Continuous tracking of Hamiltonian energy decay and basis probability convergence toward the ground state."/>
        
        <div className="analytics-metrics">
          <div className="metric-chip">
            <span>INITIAL ENERGY ⟨H₀⟩/⟨HP⟩</span>
            <strong>{history.length > 0 ? history[0].energyExpectation.toFixed(3) : "—"}</strong>
            <small>At s = 0.000 (uniform superposition)</small>
          </div>
          <div className="metric-chip">
            <span>CURRENT / FINAL ENERGY</span>
            <strong className={status==="complete" ? "green" : ""}>{frame ? frame.energyExpectation.toFixed(3) : "—"}</strong>
            <small>At s = {(frame?.s ?? 0).toFixed(3)} ({frame ? `${frame.progress}%` : "0%"})</small>
          </div>
          <div className="metric-chip">
            <span>EXACT GROUND ENERGY E₀</span>
            <strong className="green">{verification.best ** 2}</strong>
            <small>Minimum squared imbalance (diff = {verification.best})</small>
          </div>
          <div className="metric-chip">
            <span>OPTIMAL PROBABILITY</span>
            <strong className={frame && frame.optimalProbability > 0.5 ? "green" : ""}>{frame ? `${(frame.optimalProbability * 100).toFixed(2)}%` : "—"}</strong>
            <small>{verification.optimal.length} symmetric target states</small>
          </div>
        </div>

        <div className="analytics-grid">
          <div className="chart-card">
            <div className="chart-card-header">
              <div>
                <h3>Energy Expectation Trajectory</h3>
                <p>⟨HP(s)⟩ plotted continuously against adiabatic evolution progress s ∈ [0, 1]</p>
              </div>
              <div className="chart-legend">
                <span className="legend-item"><span className="legend-line energy"/>⟨HP⟩ Energy</span>
                <span className="legend-item"><span className="legend-line prob"/>Optimal P(s)</span>
              </div>
            </div>
            <EnergyGraph history={history} minEnergy={frame?.minEnergy ?? 0}/>
            <div className="chart-card-footer">
              <span>X-axis: Evolution progress s (0 → 1)</span>
              <span>Left Y: ⟨HP⟩ · Right Y: Optimal P(s)</span>
            </div>
          </div>

          <div className="chart-card">
            <div className="chart-card-header">
              <div>
                <h3>Measurement Probability Distribution</h3>
                <p>Most probable computational basis bitstrings with optimal states highlighted</p>
              </div>
              <div className="chart-legend">
                <span className="legend-item"><span className="legend-badge optimal"/>Optimal (Δ={verification.best})</span>
                <span className="legend-item"><span className="legend-badge non-optimal"/>Sub-optimal</span>
              </div>
            </div>
            <ProbabilityChart topStates={frame?.topStates} minEnergy={frame?.minEnergy ?? 0} frame={frame}/>
            <div className="chart-card-footer">
              <span>Search space: 2^{weights.length} = {1 << weights.length} states</span>
              <span>Optimal subspace: {frame ? `${(frame.optimalProbability * 100).toFixed(2)}%` : "—"}</span>
            </div>
          </div>
        </div>
      </section>}

      {tab==="simulator" && <div className="single-panel panel">
        <PanelTitle number="02" title="Simulation controls" subtitle="Start the numerical evolution to update the statevector and measurement probabilities."/>
        <div className="sim-explainer"><Atom size={30}/><div><strong>What is evolving?</strong><p>A complex statevector of 2ⁿ amplitudes. Each time step applies the problem-Hamiltonian phase and the initial-Hamiltonian mixing operation. The plotted values come from the computed statevector.</p></div></div>
        <div className="big-progress"><div><span>Evolution progress</span><b>{frame?.progress ?? 0}%</b></div><div className="progress-track"><div className="progress-fill" style={{width:`${frame?.progress||0}%`}}/></div></div>
        <div className="qubit-large-grid">{weights.map((w,i)=><div className="qubit-card" key={i}><span>QUBIT {i+1}</span><div className="qubit-orb"><div className="orb-core">{frame?(frame.probabilities.reduce((sum,p,idx)=>sum+(frame.bitStrings[idx][i]==="1"?p:0),0)>.5?"1":"0"):"+"}</div><div className="orb-ring"/></div><small>Workload {w}</small></div>)}</div>
        <div className="action-row"><button className="primary-btn" disabled={status==="running"} onClick={run}><Play size={16} fill="currentColor"/>Run evolution</button><button className="secondary-btn" onClick={reset}><RotateCcw size={16}/>Reset</button></div>
      </div>}

      {tab==="hamiltonian" && <div className="single-panel panel">
        <PanelTitle number="03" title="Ising Hamiltonian" subtitle="The objective function is encoded as a diagonal operator in the computational basis."/>
        <div className="equation-card"><span>PROBLEM HAMILTONIAN</span><div className="equation">H<sub>P</sub> = ( Σ a<sub>i</sub>Z<sub>i</sub> )<sup>2</sup></div><p>Its energy is the squared difference between the two server loads. The ground-state energy is the minimum possible squared imbalance.</p></div>
        <div className="equation-card"><span>INITIAL HAMILTONIAN</span><div className="equation">H<sub>0</sub> = − Σ X<sub>i</sub></div><p>The ground state is an equal superposition of all computational basis assignments.</p></div>
        <div className="equation-card"><span>ADIABATIC INTERPOLATION</span><div className="equation">H(s) = (1 − s)H<sub>0</sub> + sH<sub>P</sub></div><p>The parameter s moves from 0 to 1. The simulation approximates time evolution with small discrete steps.</p></div>
        <div className="note-box"><CircleHelp size={17}/><p>The product formula is a numerical approximation. Finite-time evolution does not guarantee that the final measurement is optimal; use the probability chart and independent brute-force check.</p></div>
      </div>}

      {tab==="verification" && <div className="single-panel panel">
        <PanelTitle number="04" title="Classical verification" subtitle="Exhaustively evaluate every possible assignment, independent of the quantum simulation."/>
        <div className="verification-summary"><div><span>Assignments checked</span><strong>{verification.rows.length}</strong></div><div><span>Minimum difference</span><strong className="green">{verification.best}</strong></div><div><span>Optimal bitstrings</span><strong>{verification.optimal.length}</strong></div></div>
        <div className="table-wrap"><table><thead><tr><th>Bitstring</th><th>Server A</th><th>Server B</th><th>Difference</th><th>Result</th></tr></thead><tbody>
          {(showAll?verification.rows:verification.rows.filter(r=>r.diff===verification.best).concat(verification.rows.filter(r=>r.diff!==verification.best).slice(0,8))).map(r=><tr key={r.bits} className={r.diff===verification.best?"optimal-row":""}><td><code>{r.bits}</code></td><td>{r.a}</td><td>{r.b}</td><td>{r.diff}</td><td>{r.diff===verification.best?<span className="optimal-tag">OPTIMAL</span>:"—"}</td></tr>)}
        </tbody></table></div>
        <button className="secondary-btn" onClick={()=>setShowAll(!showAll)}>{showAll?"Show fewer rows":`Show all ${verification.rows.length} assignments`}</button>
      </div>}

      {tab==="overview" && <section className="panel results-panel">
        <PanelTitle number="04" title="Server allocation" subtitle="A candidate assignment from the quantum measurement distribution, compared with the exact optimum."/>
        <div className="allocation-grid">
          <ServerCard title="SERVER A" color="purple" bits={allocation?.bits || null} weights={weights} server="1" load={allocation?.a ?? null} fallback={verification.optimal[0]?.a}/>
          <ServerCard title="SERVER B" color="green" bits={allocation?.bits || null} weights={weights} server="0" load={allocation?.b ?? null} fallback={verification.optimal[0]?.b}/>
        </div>
        <div className="result-footer"><div><span>Quantum candidate</span><strong>{chosen?.bits || "Run simulation to generate"}</strong></div><div><span>Candidate imbalance</span><strong>{allocation?allocation.diff:"—"}</strong></div><div><span>Exact optimum</span><strong className="green">{verification.best}</strong></div></div>
        {status==="complete" && <div className={`notice ${allocation?.diff===verification.best?"success":"warning"}`}>{allocation?.diff===verification.best?"The most probable displayed candidate is optimal.":"The displayed candidate is not optimal; the exact brute-force result remains the reference."} Optimal-subspace probability: {(optimalProbability*100).toFixed(2)}%.</div>}
      </section>}

      <footer><span>Q-BALANCE · EDUCATIONAL QUANTUM SIMULATION</span><span>STATEVECTOR · ADIABATIC EVOLUTION · BRUTE FORCE</span></footer>
    </main>
  </div>;
}

function Stat({icon,label,value,note,accent}) { return <div className="stat-card"><div className="stat-top"><span>{label}</span><span className="stat-icon">{icon}</span></div><strong className={accent?"green":""}>{value}</strong><small>{note}</small></div>; }
function PanelTitle({number,title,subtitle}) { return <div className="panel-title"><div className="section-number">{number}</div><div><h2>{title}</h2><p>{subtitle}</p></div></div>; }
function ServerCard({title,color,bits,weights,server,load,fallback}) {
  const assigned = bits ? weights.filter((_,i)=>bits[i]===server) : [];
  const value = load ?? fallback;
  return <div className={`server-card ${color}`}><div className="server-heading"><span>{title}</span><span className="server-indicator"/></div><div className="server-load">{value ?? "—"}<small>load units</small></div><div className="server-items">{assigned.length?assigned.map((w,i)=><span key={i}>{w}</span>):<span className="placeholder">{bits?"No items assigned":"Awaiting simulation"}</span>}</div><div className="server-foot">{bits?`${assigned.length} assigned items`:"Exact optimal load preview"}</div></div>;
}

createRoot(document.getElementById("root")).render(<App/>);
