/**
 * Headless benchmark: ms per physics step at several body counts.
 *
 * No rendering, no DOM — pure physics on the seeded disk-galaxy scenario,
 * so results are reproducible and become the before/after baseline for the
 * Phase 2 Barnes–Hut comparison. Run with `npm run bench`.
 *
 * The "max bodies at 60 fps" claim in the README is derived from this
 * table: the largest benchmarked N whose ms/step fits the 16.67 ms frame
 * budget (one physics step per frame).
 */
import { Leapfrog } from '../src/physics/leapfrog';
import { diskGalaxy } from '../src/scenarios/disk-galaxy';

const SEED = 1;
const FRAME_BUDGET_MS = 1000 / 60;

/** Step counts sized so each row does comparable total work (N² · steps). */
const RUNS = [
  { n: 500, steps: 400 },
  { n: 1000, steps: 200 },
  { n: 2000, steps: 100 },
  { n: 4000, steps: 50 },
];

interface Row {
  n: number;
  msPerStep: number;
}

function timeRun(n: number, steps: number): number {
  const state = diskGalaxy.generate(n, SEED);
  const integrator = new Leapfrog();
  integrator.init(state);

  // Warm up the JIT before timing.
  const warmup = Math.max(5, Math.round(steps / 10));
  for (let i = 0; i < warmup; i++) integrator.step(state, diskGalaxy.dt);

  // Best of 3 repetitions to shrug off scheduler noise.
  let best = Infinity;
  for (let rep = 0; rep < 3; rep++) {
    const t0 = performance.now();
    for (let i = 0; i < steps; i++) integrator.step(state, diskGalaxy.dt);
    best = Math.min(best, (performance.now() - t0) / steps);
  }
  return best;
}

const rows: Row[] = RUNS.map(({ n, steps }) => {
  const msPerStep = timeRun(n, steps);
  console.error(`  n=${n}: ${msPerStep.toFixed(3)} ms/step`);
  return { n, msPerStep };
});

console.log('\n| Bodies | ms / step | Steps / frame budget (16.7 ms) |');
console.log('|-------:|----------:|-------------------------------:|');
for (const { n, msPerStep } of rows) {
  console.log(
    `| ${n.toLocaleString('en-US')} | ${msPerStep.toFixed(3)} | ${(FRAME_BUDGET_MS / msPerStep).toFixed(1)} |`,
  );
}

const sustainable = rows.filter((r) => r.msPerStep <= FRAME_BUDGET_MS);
if (sustainable.length > 0) {
  const max = sustainable[sustainable.length - 1];
  console.log(
    `\nMax benchmarked body count within the 60 fps budget: ` +
      `${max.n.toLocaleString('en-US')} (${max.msPerStep.toFixed(3)} ms/step).`,
  );
} else {
  console.log('\nNo benchmarked body count fits the 60 fps budget.');
}
