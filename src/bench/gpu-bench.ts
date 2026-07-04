import { acquireGpu, type GpuContext } from '../gpu/device';
import { GpuLeapfrog } from '../gpu/gpu-leapfrog';
import { BarnesHut } from '../physics/barnes-hut';
import type { Integrator } from '../physics/integrator';
import { Leapfrog } from '../physics/leapfrog';
import { diskGalaxy } from '../scenarios/disk-galaxy';

/**
 * Browser-run GPU benchmark (/bench.html): ms per physics step, Float32
 * WebGPU brute force vs CPU Barnes–Hut in the SAME page on the SAME
 * machine — the fair Phase 3 crossover comparison. `npm run bench` (Node)
 * remains the CPU-only Phase 2 benchmark; Node cannot time WebGPU.
 *
 * Methodology carried over from scripts/bench.ts, honesty rules included:
 * θ = 0.5 is THE benchmark setting; any method at any size is skipped —
 * never extrapolated — once a probe step exceeds BRUTE-style cutoff; the
 * max-N-at-60fps claim is the largest measured N under the frame budget.
 * GPU timing submits a batch of steps and waits for onSubmittedWorkDone,
 * so reported numbers are wall-clock throughput including submit overhead,
 * not encode time. The GPU adapter is named in the output — GPU numbers
 * mean nothing without the hardware next to them.
 *
 * The physics step is timed WITHOUT rendering on purpose: Canvas 2D
 * cannot draw 10⁵ sprites per frame, so at GPU-scale N the renderer, not
 * the kernel, is the frame-rate ceiling. That is a documented limitation
 * (see README), not something this page hides.
 */
const SIZES = [2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 500_000];
const THETA = 0.5;
const SEED = 1;
const CUTOFF_MS = 2_000;
const FRAME_BUDGET_MS = 1000 / 60;
/** Sizing target for one timed repetition (3 reps, best-of). */
const REP_TARGET_MS = 400;

const logEl = document.getElementById('log') as HTMLPreElement;
function log(line: string): void {
  logEl.textContent += `${line}\n`;
  console.log(`[bench] ${line}`);
}

/** Let the browser paint the log line before blocking again. */
const breathe = (): Promise<void> => new Promise((r) => setTimeout(r, 0));

interface Row {
  n: number;
  bh: number | null; // ms/step, null = skipped (probe exceeded cutoff)
  gpu: number | null;
}

function timeCpuBh(n: number): number | null {
  const bh = new BarnesHut(THETA);
  const state = diskGalaxy.generate(n, SEED);
  const integrator: Integrator = new Leapfrog(bh.accelerations);
  integrator.init(state);

  const probe0 = performance.now();
  integrator.step(state, diskGalaxy.dt);
  const probe = performance.now() - probe0;
  if (probe > CUTOFF_MS) return null;

  const steps = Math.max(1, Math.round(REP_TARGET_MS / probe));
  const warmup = Math.max(3, Math.round(steps / 10));
  for (let i = 0; i < warmup; i++) integrator.step(state, diskGalaxy.dt);
  let best = Infinity;
  for (let rep = 0; rep < 3; rep++) {
    const t0 = performance.now();
    for (let i = 0; i < steps; i++) integrator.step(state, diskGalaxy.dt);
    best = Math.min(best, (performance.now() - t0) / steps);
  }
  return best;
}

async function timeGpu(ctx: GpuContext, gpu: GpuLeapfrog, n: number): Promise<number | null> {
  const state = diskGalaxy.generate(n, SEED);
  gpu.init(state);

  // Warm up pipelines/driver before probing, then measure one synced step.
  for (let i = 0; i < 3; i++) gpu.step(state, diskGalaxy.dt);
  await ctx.device.queue.onSubmittedWorkDone();
  const probe0 = performance.now();
  gpu.step(state, diskGalaxy.dt);
  await ctx.device.queue.onSubmittedWorkDone();
  const probe = performance.now() - probe0;
  if (probe > CUTOFF_MS) return null;

  const steps = Math.min(400, Math.max(1, Math.round(REP_TARGET_MS / probe)));
  let best = Infinity;
  for (let rep = 0; rep < 3; rep++) {
    const t0 = performance.now();
    for (let i = 0; i < steps; i++) gpu.step(state, diskGalaxy.dt);
    await ctx.device.queue.onSubmittedWorkDone();
    best = Math.min(best, (performance.now() - t0) / steps);
  }
  return best;
}

async function run(): Promise<void> {
  const ctx = await acquireGpu();
  if (ctx === null) {
    log('No WebGPU adapter — the GPU benchmark cannot run in this browser.');
    document.body.dataset.benchDone = '1';
    return;
  }
  ctx.lost.then((info) => log(`!! GPU device lost mid-benchmark (${info.reason}): ${info.message}`));

  log(`GPU adapter: ${ctx.adapterLabel}`);
  log(`User agent: ${navigator.userAgent}`);
  log(`Scenario: seeded disk galaxy, dt=${diskGalaxy.dt}, θ=${THETA} (CPU Barnes–Hut)`);
  log('');

  const gpu = new GpuLeapfrog(ctx.device);
  const rows: Row[] = [];
  let bhAlive = true;
  let gpuAlive = true;
  for (const n of SIZES) {
    await breathe();
    let bh: number | null = null;
    if (bhAlive) {
      bh = timeCpuBh(n);
      if (bh === null) bhAlive = false;
    }
    await breathe();
    let gpuMs: number | null = null;
    if (gpuAlive) {
      gpuMs = await timeGpu(ctx, gpu, n);
      if (gpuMs === null) gpuAlive = false;
    }
    rows.push({ n, bh, gpu: gpuMs });
    log(
      `  n=${n.toLocaleString('en-US')}: ` +
        `bh=${bh === null ? 'skipped' : `${bh.toFixed(2)} ms`}, ` +
        `gpu=${gpuMs === null ? 'skipped' : `${gpuMs.toFixed(2)} ms`}`,
    );
  }

  log('');
  log(`| Bodies | CPU BH (θ=${THETA}) ms/step | GPU brute f32 ms/step | GPU speedup |`);
  log('|-------:|--------------------------:|----------------------:|------------:|');
  for (const { n, bh, gpu: g } of rows) {
    const bhCell = bh === null ? '— (> 2 s, skipped)' : bh.toFixed(2);
    const gpuCell = g === null ? '— (> 2 s, skipped)' : g.toFixed(2);
    const speedup = bh === null || g === null ? '—' : `${(bh / g).toFixed(1)}×`;
    log(`| ${n.toLocaleString('en-US')} | ${bhCell} | ${gpuCell} | ${speedup} |`);
  }

  const crossover = rows.find((r) => r.bh !== null && r.gpu !== null && r.gpu < r.bh);
  log('');
  log(
    crossover
      ? `Crossover: GPU brute force wins from N = ${crossover.n.toLocaleString('en-US')} up.`
      : 'No crossover measured: CPU Barnes–Hut wins throughout the measured range.',
  );
  for (const [label, pick] of [
    ['CPU Barnes–Hut', (r: Row): number | null => r.bh],
    ['GPU brute force', (r: Row): number | null => r.gpu],
  ] as const) {
    const fits = rows.filter((r) => {
      const ms = pick(r);
      return ms !== null && ms <= FRAME_BUDGET_MS;
    });
    const max = fits[fits.length - 1];
    log(
      max === undefined
        ? `Max N at 60 fps (${label}): none in the measured range.`
        : `Max N at 60 fps (${label}): ${max.n.toLocaleString('en-US')} ` +
            `(${(pick(max) as number).toFixed(2)} ms/step, physics only — see README on rendering).`,
    );
  }
  document.body.dataset.benchDone = '1';
}

const button = document.getElementById('run') as HTMLButtonElement;
button.addEventListener('click', () => {
  button.disabled = true;
  void run();
});
if (new URLSearchParams(location.search).has('auto')) {
  button.disabled = true;
  void run();
}
