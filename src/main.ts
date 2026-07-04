import { BarnesHut } from './physics/barnes-hut';
import { relativeEnergyDrift } from './physics/diagnostics';
import { EnergySampler } from './physics/energy-sampler';
import { computeAccelerations, type ForceCalculator } from './physics/forces';
import { Leapfrog } from './physics/leapfrog';
import { Simulation } from './physics/simulation';
import type { ForceMethod } from './scenarios/types';
import { attachCameraControls, Camera } from './render/camera';
import { Renderer } from './render/renderer';
import { getScenario, scenarios } from './scenarios/registry';
import { buildControls } from './ui/controls';
import { StatsOverlay } from './ui/stats';

/**
 * Fixed scenario seed: every page load and every Reset produce the same
 * system, so behaviour is comparable across runs and code changes.
 */
const SEED = 42;

/**
 * Overlay refresh cadence, and the minimum gap between energy samples.
 * The energy sample is O(N²) no matter which force method runs (the
 * potential must use the exact softened kernel — see forces.ts), so it is
 * computed by the time-sliced EnergySampler a few ms per frame instead of
 * in one freezing pass; the gap between samples adapts to the measured
 * sample cost (see the frame loop).
 */
const STATS_INTERVAL_S = 0.5;

/** Keep the amortized energy-sampling cost below ~5% of wall-clock time. */
const STATS_COST_FACTOR = 20;

/** Per-frame slice of energy-sampling work (~1/5 of a 60 fps frame). */
const SAMPLE_BUDGET_MS = 3;

function el<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) throw new Error(`Missing #${id}`);
  return node as T;
}

let scenario = getScenario('disk-galaxy');
let count = scenario.defaultN;

/**
 * Force-method dispatch. The integrator holds one ForceCalculator for its
 * lifetime, so the closure switches on the current UI selection instead of
 * rebuilding the integrator. Switching mid-run is safe: the next step's
 * opening kick reuses accelerations from the previous method — a one-off
 * O(θ)-sized nudge, same order as the approximation itself.
 */
const barnesHut = new BarnesHut();
let forceMethod: ForceMethod = scenario.forceMethod ?? 'brute';
const force: ForceCalculator = (s) =>
  forceMethod === 'barnes-hut' ? barnesHut.accelerations(s) : computeAccelerations(s);

const sim = new Simulation(scenario.generate(count, SEED), new Leapfrog(force), scenario.dt);

/**
 * Energy diagnostics run through the time-sliced sampler: 'baseline'
 * establishes E₀ after every (re)generation, then 'drift' samples repeat on
 * an adaptive gap. drift stays null until the first full E₀+E pair exists.
 */
const sampler = new EnergySampler();
let e0: number | null = null;
let drift: number | null = null;
let sampleKind: 'baseline' | 'drift' | null = null;
let sampleGap = STATS_INTERVAL_S;
let sinceSample = 0;

function beginBaseline(): void {
  e0 = null;
  drift = null;
  sampler.begin(sim.state);
  sampleKind = 'baseline';
}
beginBaseline();

const canvas = el<HTMLCanvasElement>('sim-canvas');
const renderer = new Renderer(canvas);
const camera = new Camera();
const applyViewport = (): void => {
  const { width, height } = renderer.resize();
  camera.setViewport(width, height);
};
applyViewport();
camera.fit(sim.state);
attachCameraControls(canvas, camera);
window.addEventListener('resize', applyViewport);

const stats = new StatsOverlay(el('stats'));

function resetScenario(dt: number): void {
  sim.reset(scenario.generate(count, SEED), dt);
  beginBaseline();
  camera.fit(sim.state);
}

const controls = buildControls(el('controls'), {
  scenarios,
  activeId: scenario.id,
  count,
  dt: sim.dt,
  forceMethod,
  theta: barnesHut.theta,
  onTogglePlay: () => (sim.running = !sim.running),
  onReset: () => resetScenario(sim.dt),
  onScenarioChange: (id) => {
    scenario = getScenario(id);
    count = scenario.defaultN;
    forceMethod = scenario.forceMethod ?? 'brute';
    resetScenario(scenario.dt);
    controls.syncScenario(scenario, count, sim.dt, forceMethod);
  },
  onCountChange: (n) => {
    // A new body count can't apply to a live system — regenerate.
    count = n;
    resetScenario(sim.dt);
  },
  onDtChange: (dt) => {
    // Applies live; see the slider tooltip for the symplectic caveat.
    sim.dt = dt;
  },
  onForceMethodChange: (method) => {
    forceMethod = method;
  },
  onThetaChange: (theta) => {
    barnesHut.theta = theta;
  },
});

let last = performance.now();
let statsClock = 0;
let framesSinceStats = 0;

function frame(now: number): void {
  const elapsed = (now - last) / 1000;
  last = now;

  sim.advance(elapsed);
  renderer.draw(sim.state, camera);

  // Energy sampling: pump the in-progress sample a slice per frame, or
  // schedule the next one. The gap adapts to the measured sample cost so
  // the amortized overhead stays ~1/STATS_COST_FACTOR of wall time no
  // matter how large N² grows.
  if (sampleKind !== null) {
    const e = sampler.tick(SAMPLE_BUDGET_MS);
    if (e !== null) {
      if (sampleKind === 'baseline') {
        e0 = e;
      } else if (e0 !== null) {
        drift = relativeEnergyDrift(e, e0);
      }
      sampleKind = null;
      sinceSample = 0;
      sampleGap = Math.max(STATS_INTERVAL_S, (STATS_COST_FACTOR * sampler.lastCostMs) / 1000);
    }
  } else {
    sinceSample += elapsed;
    if (sinceSample >= sampleGap) {
      sampler.begin(sim.state);
      sampleKind = e0 === null ? 'baseline' : 'drift';
    }
  }

  framesSinceStats++;
  statsClock += elapsed;
  if (statsClock >= STATS_INTERVAL_S) {
    stats.update({
      fps: framesSinceStats / statsClock,
      drift,
      sampling: sampleKind !== null,
      bodies: sim.state.n,
      paused: !sim.running,
    });
    framesSinceStats = 0;
    statsClock = 0;
  }
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
