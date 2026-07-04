import { beforeAll, describe, expect, it } from 'vitest';
import {
  momentumScale,
  relativeEnergyDrift,
  totalEnergy,
  totalMomentum,
} from '../physics/diagnostics';
import { Leapfrog } from '../physics/leapfrog';
import { diskGalaxy } from '../scenarios/disk-galaxy';
import { GpuLeapfrog } from './gpu-leapfrog';
import { acquireGpuForTests, stepMany } from './gpu-test-utils';

const ctx = await acquireGpuForTests();

/**
 * Conservation contracts of the Float32 GPU integrator on the pinned disk
 * trajectory, with the Float64 CPU run alongside — the headline Phase 3
 * numbers the README reports.
 *
 * PINNED SETUP — do not loosen tolerances or reshuffle the seed. GPU float
 * math is not bit-reproducible across vendors, so bounds are
 * measured-then-pinned (the Phase 2 momentum-bound precedent) with ~10×
 * headroom for reduction-order differences. Measured on Apple M-series
 * (metal-3), Chromium 149:
 *
 * - Energy, 1,000 steps: cpu-f64 1.35e-6 vs gpu-f32 1.48e-6 (1.1×).
 * - Energy, 10,000 steps: cpu-f64 1.34e-6 vs gpu-f32 1.25e-6 (0.9×).
 *   The headline finding: Float32 energy drift is INDISTINGUISHABLE from
 *   Float64 on this trajectory. Leapfrog's bounded symplectic oscillation
 *   (set by dt² truncation) dominates both runs, and f32 rounding noise
 *   stays below that band even after 10k steps — the drift diagnostic
 *   cannot tell the two apart until far longer horizons.
 * - Momentum: the CPU brute path conserves |p| to 1e-12 of the momentum
 *   scale because pair terms cancel EXACTLY (Newton's third law applied
 *   to the same f64 numbers). The GPU kernel sums each body independently,
 *   so cancellation holds only to f32 rounding — machine-precision
 *   momentum does NOT survive the port, by design. Measured after 1,000
 *   steps: 4.6e-8 of the momentum scale.
 */
const N = 1024; // disk particles; the scenario adds the central body
const SEED = 42;
const DT = 0.002;
const STEPS = 1000;
const STEPS_LONG = 10000;

const ENERGY_TOL = 2e-5;
const ENERGY_TOL_LONG = 2e-5;
const MOMENTUM_TOL = 5e-7;

describe.skipIf(ctx === null)('GPU integrator conservation (pinned disk)', () => {
  let gpuDrift = Number.NaN;
  let gpuDriftLong = Number.NaN;
  let cpuDrift = Number.NaN;
  let cpuDriftLong = Number.NaN;
  let momentumRel = Number.NaN;

  beforeAll(async () => {
    // GPU Float32 run. E₀ is taken from the quantized initial state (init
    // writes it back into the mirror), so drift measures integration, not
    // the one-off upload rounding.
    const gpuState = diskGalaxy.generate(N, SEED);
    const gpu = new GpuLeapfrog(ctx!.device);
    gpu.init(gpuState);
    const e0 = totalEnergy(gpuState);
    await stepMany(ctx!, gpu, gpuState, DT, STEPS);
    await gpu.readState(gpuState);
    gpuDrift = relativeEnergyDrift(totalEnergy(gpuState), e0);
    momentumRel = totalMomentum(gpuState).mag / momentumScale(gpuState);
    await stepMany(ctx!, gpu, gpuState, DT, STEPS_LONG - STEPS);
    await gpu.readState(gpuState);
    gpuDriftLong = relativeEnergyDrift(totalEnergy(gpuState), e0);

    // CPU Float64 run, same pinned trajectory — the README comparison.
    const cpuState = diskGalaxy.generate(N, SEED);
    const cpu = new Leapfrog();
    cpu.init(cpuState);
    const e0Cpu = totalEnergy(cpuState);
    for (let s = 0; s < STEPS; s++) cpu.step(cpuState, DT);
    cpuDrift = relativeEnergyDrift(totalEnergy(cpuState), e0Cpu);
    for (let s = 0; s < STEPS_LONG - STEPS; s++) cpu.step(cpuState, DT);
    cpuDriftLong = relativeEnergyDrift(totalEnergy(cpuState), e0Cpu);

    console.log(
      `pinned drift (N=${gpuState.n}, dt=${DT}):\n` +
        `  ${STEPS} steps: cpu-f64=${cpuDrift.toExponential(2)} ` +
        `gpu-f32=${gpuDrift.toExponential(2)} (${(gpuDrift / cpuDrift).toFixed(1)}×)\n` +
        `  ${STEPS_LONG} steps: cpu-f64=${cpuDriftLong.toExponential(2)} ` +
        `gpu-f32=${gpuDriftLong.toExponential(2)} (${(gpuDriftLong / cpuDriftLong).toFixed(1)}×)\n` +
        `  gpu momentum after ${STEPS} steps: ${momentumRel.toExponential(2)} of scale`,
    );
  });

  it('keeps 1k-step energy drift at the Float64 scale (truncation-dominated)', () => {
    expect(gpuDrift).toBeLessThan(ENERGY_TOL);
  });

  it('keeps 10k-step energy drift bounded as f32 rounding accumulates', () => {
    expect(gpuDriftLong).toBeLessThan(ENERGY_TOL_LONG);
  });

  it('drifts momentum at f32 rounding scale (bounded, not machine-zero)', () => {
    expect(momentumRel).toBeLessThan(MOMENTUM_TOL);
  });
});
