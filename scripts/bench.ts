/**
 * Headless benchmark: ms per physics step, brute force vs Barnes-Hut, at
 * several body counts on the seeded disk-galaxy scenario.
 *
 * No rendering, no DOM — results are reproducible and feed the README
 * table directly (update the README by re-running this, never by editing
 * numbers). Run with `npm run bench`.
 *
 * Honesty rules baked in:
 * - theta = 0.5 is THE benchmark setting — no tuning it up for the table.
 * - Brute force is measured at every size until a single step exceeds
 *   BRUTE_CUTOFF_MS (one probe step decides); larger sizes are skipped,
 *   not extrapolated.
 * - The "max N at 60 fps" claim is the largest benchmarked N whose
 *   Barnes-Hut step fits the 16.67 ms frame budget, nothing fancier.
 *
 * The allocation-discipline check at the end samples heap usage across a
 * long Barnes-Hut run: the tree arena is pre-allocated and reused, so
 * steady-state heap growth should be ~zero. A leak (per-step node objects,
 * closures in the hot path) would show up as monotonic growth here.
 */
import { BarnesHut } from '../src/physics/barnes-hut';
import { computeAccelerations, type ForceCalculator } from '../src/physics/forces';
import { Leapfrog } from '../src/physics/leapfrog';
import { diskGalaxy } from '../src/scenarios/disk-galaxy';

/**
 * The benchmark runs under Node (tsx), but the project compiles against
 * DOM lib types; declare the one Node global we use instead of pulling in
 * @types/node for a single script.
 */
declare const process: { memoryUsage(): { heapUsed: number } };

const SEED = 1;
const THETA = 0.5;
const FRAME_BUDGET_MS = 1000 / 60;
const BRUTE_CUTOFF_MS = 2000;

const SIZES = [1000, 2000, 5000, 10000, 20000, 50000];

/**
 * Steps per timed repetition, sized so each rep does comparable total work
 * (~N² pair-ops for brute, ~N steps of tree work for BH) and no single rep
 * runs away at large N.
 */
const BRUTE_STEPS: Record<number, number> = {
  1000: 100,
  2000: 50,
  5000: 10,
  10000: 4,
  20000: 2,
  50000: 1,
};
const BH_STEPS: Record<number, number> = {
  1000: 200,
  2000: 100,
  5000: 50,
  10000: 25,
  20000: 12,
  50000: 5,
};

/** Best-of-3 ms/step for one (n, force) pair, with JIT warmup. */
function timeRun(n: number, steps: number, force: ForceCalculator): number {
  const state = diskGalaxy.generate(n, SEED);
  const integrator = new Leapfrog(force);
  integrator.init(state);

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

/** One probe step of brute force — decides whether full timing is sane. */
function bruteProbe(n: number): number {
  const state = diskGalaxy.generate(n, SEED);
  const integrator = new Leapfrog();
  integrator.init(state);
  const t0 = performance.now();
  integrator.step(state, diskGalaxy.dt);
  return performance.now() - t0;
}

interface Row {
  n: number;
  brute: number | null; // null = skipped (probe step exceeded the cutoff)
  bh: number;
}

let bruteAlive = true;
const rows: Row[] = SIZES.map((n) => {
  let brute: number | null = null;
  if (bruteAlive && bruteProbe(n) <= BRUTE_CUTOFF_MS) {
    brute = timeRun(n, BRUTE_STEPS[n], computeAccelerations);
  } else {
    bruteAlive = false;
  }
  const bh = timeRun(n, BH_STEPS[n], new BarnesHut(THETA).accelerations);
  console.error(
    `  n=${n}: brute=${brute === null ? 'skipped' : `${brute.toFixed(2)} ms`}, ` +
      `bh(θ=${THETA})=${bh.toFixed(2)} ms`,
  );
  return { n, brute, bh };
});

console.log(`\n| Bodies | Brute ms/step | BH (θ=${THETA}) ms/step | Speedup |`);
console.log('|-------:|--------------:|--------------------:|--------:|');
for (const { n, brute, bh } of rows) {
  const bruteCell = brute === null ? '— (> 2 s, skipped)' : brute.toFixed(2);
  const speedup = brute === null ? '—' : `${(brute / bh).toFixed(1)}×`;
  console.log(`| ${n.toLocaleString('en-US')} | ${bruteCell} | ${bh.toFixed(2)} | ${speedup} |`);
}

const crossover = rows.find((r) => r.brute !== null && r.bh < r.brute);
console.log(
  crossover
    ? `\nCrossover: Barnes-Hut wins from N = ${crossover.n.toLocaleString('en-US')} up.`
    : '\nNo crossover in the benchmarked range: brute force wins throughout.',
);

const sustainable = rows.filter((r) => r.bh <= FRAME_BUDGET_MS);
if (sustainable.length > 0) {
  const max = sustainable[sustainable.length - 1];
  console.log(
    `Max benchmarked body count within the 60 fps budget (Barnes-Hut): ` +
      `${max.n.toLocaleString('en-US')} (${max.bh.toFixed(2)} ms/step).`,
  );
} else {
  console.log('No benchmarked body count fits the 60 fps budget with Barnes-Hut.');
}

// --- allocation-discipline check ------------------------------------------
{
  const n = 20000;
  const state = diskGalaxy.generate(n, SEED);
  const bh = new BarnesHut(THETA);
  const integrator = new Leapfrog(bh.accelerations);
  integrator.init(state);
  for (let i = 0; i < 30; i++) integrator.step(state, diskGalaxy.dt); // warm arena + JIT
  const samples: number[] = [];
  for (let block = 0; block < 5; block++) {
    for (let i = 0; i < 60; i++) integrator.step(state, diskGalaxy.dt);
    samples.push(process.memoryUsage().heapUsed / 1024 / 1024);
  }
  const spread = Math.max(...samples) - Math.min(...samples);
  console.log(
    `\nAllocation check: heap over 300 BH steps at N=${n.toLocaleString('en-US')}: ` +
      `[${samples.map((s) => s.toFixed(1)).join(', ')}] MB (spread ${spread.toFixed(2)} MB).`,
  );
}
