import type { SimState } from '../physics/state';
import { acquireGpu, type GpuContext } from './device';
import type { GpuLeapfrog } from './gpu-leapfrog';

/**
 * Shared plumbing for the *.gpu.test.ts suites (browser mode only — see
 * vitest.gpu.config.ts).
 *
 * No adapter (a CI box without a GPU, a browser without WebGPU) must skip
 * LOUDLY, never silently green: every suite hangs off describe.skipIf on
 * the result of this call, and the warning below is the marker to grep
 * for when a "passing" GPU run looks suspiciously fast.
 */
export async function acquireGpuForTests(): Promise<GpuContext | null> {
  const ctx = await acquireGpu();
  if (ctx === null) {
    console.warn(
      '⚠️  No WebGPU adapter available — the GPU test suite is being SKIPPED, not passed.',
    );
  } else {
    console.log(`GPU test adapter: ${ctx.adapterLabel}`);
  }
  return ctx;
}

/**
 * Drive `steps` GPU leapfrog steps, syncing with the device every couple
 * hundred submits so the command queue stays shallow on long runs.
 */
export async function stepMany(
  ctx: GpuContext,
  gpu: GpuLeapfrog,
  state: SimState,
  dt: number,
  steps: number,
): Promise<void> {
  for (let s = 1; s <= steps; s++) {
    gpu.step(state, dt);
    if (s % 200 === 0) await ctx.device.queue.onSubmittedWorkDone();
  }
}
