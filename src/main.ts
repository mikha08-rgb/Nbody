import { BarnesHut } from './physics/barnes-hut';
import { relativeEnergyDrift, totalEnergy } from './physics/diagnostics';
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
 * Minimum wall-clock interval between diagnostics samples. The energy
 * sample is O(N²) no matter which force method runs (the potential must
 * use the exact softened kernel — see forces.ts), so at Barnes-Hut body
 * counts a fixed interval would freeze the app. The actual interval
 * adapts to the measured sample cost (see the frame loop).
 */
const STATS_INTERVAL_S = 0.5;

/** Keep the energy sample below ~5% of wall-clock time. */
const STATS_COST_FACTOR = 20;

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
let e0 = totalEnergy(sim.state);
let drift: number | null = null;

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
  e0 = totalEnergy(sim.state);
  drift = null;
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
let statsInterval = STATS_INTERVAL_S;

function frame(now: number): void {
  const elapsed = (now - last) / 1000;
  last = now;

  sim.advance(elapsed);
  renderer.draw(sim.state, camera);

  framesSinceStats++;
  statsClock += elapsed;
  if (statsClock >= statsInterval) {
    // Time the O(N²) energy sample and space the next one so sampling
    // stays a rounding error in the frame budget (~1 hitch per interval
    // at worst) instead of freezing large-N Barnes-Hut runs.
    const t0 = performance.now();
    drift = relativeEnergyDrift(totalEnergy(sim.state), e0);
    statsInterval = Math.max(
      STATS_INTERVAL_S,
      (STATS_COST_FACTOR * (performance.now() - t0)) / 1000,
    );
    stats.update({
      fps: framesSinceStats / statsClock,
      drift,
      bodies: sim.state.n,
      paused: !sim.running,
    });
    framesSinceStats = 0;
    statsClock = 0;
  }
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
