import { relativeEnergyDrift, totalEnergy } from './physics/diagnostics';
import { Leapfrog } from './physics/leapfrog';
import { Simulation } from './physics/simulation';
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

/** Wall-clock interval between diagnostics samples (energy is O(N²)). */
const STATS_INTERVAL_S = 0.5;

function el<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) throw new Error(`Missing #${id}`);
  return node as T;
}

let scenario = getScenario('disk-galaxy');
let count = scenario.defaultN;

const sim = new Simulation(scenario.generate(count, SEED), new Leapfrog(), scenario.dt);
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
  onTogglePlay: () => (sim.running = !sim.running),
  onReset: () => resetScenario(sim.dt),
  onScenarioChange: (id) => {
    scenario = getScenario(id);
    count = scenario.defaultN;
    resetScenario(scenario.dt);
    controls.syncScenario(scenario, count, sim.dt);
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
});

let last = performance.now();
let statsClock = 0;
let framesSinceStats = 0;

function frame(now: number): void {
  const elapsed = (now - last) / 1000;
  last = now;

  sim.advance(elapsed);
  renderer.draw(sim.state, camera);

  framesSinceStats++;
  statsClock += elapsed;
  if (statsClock >= STATS_INTERVAL_S) {
    drift = relativeEnergyDrift(totalEnergy(sim.state), e0);
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
