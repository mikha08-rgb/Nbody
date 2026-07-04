import { beforeAll, describe, expect, it } from 'vitest';
import { relativeEnergyDrift, totalEnergy } from '../physics/diagnostics';
import { Leapfrog } from '../physics/leapfrog';
import type { SimState } from '../physics/state';
import { figureEight } from '../scenarios/figure-eight';
import { sunPlanets } from '../scenarios/sun-planets';
import { GpuLeapfrog } from './gpu-leapfrog';
import { acquireGpuForTests, stepMany } from './gpu-test-utils';

const ctx = await acquireGpuForTests();

/**
 * The ε = 0 reality check: do the exact-solution scenarios survive
 * Float32? Measured first, then pinned — these bounds are FINDINGS, not
 * requirements; if a future change moves them, the physics of the port
 * changed and the README precision story must be re-measured, not the
 * numbers loosened.
 *
 * Method: run CPU Float64 (from the original initial conditions — the
 * trajectory any CPU user gets) and GPU Float32 (from its quantized
 * initial conditions) side by side, and compare positions at checkpoints.
 * This measures the TOTAL cost of the port — ~1e-7 initial quantization
 * plus per-step f32 rounding — which is the honest end-to-end question.
 */
describe.skipIf(ctx === null)('Float32 vs the ε = 0 exact-solution scenarios', () => {
  /**
   * Figure-eight: the choreography is only marginally stable, so Float32
   * perturbations (~1e-7 at upload, ~6e-8 per step) could plausibly grow
   * until the eight falls apart — losing it eventually would be expected
   * no matter how correct the shader is. The measured question is "after
   * how many periods".
   */
  describe('figure-eight choreography (marginally stable)', () => {
    const T = 6.32591398; // published period, see figure-eight.ts
    const STEPS_PER_PERIOD = 1266; // ≈ the scenario's default dt = 0.005
    const DT = T / STEPS_PER_PERIOD;
    const PERIODS = 40;

    const deviations: number[] = []; // max body deviation at each period

    beforeAll(async () => {
      const gpuState = figureEight.generate(3, 0);
      const cpuState = figureEight.generate(3, 0);
      const gpu = new GpuLeapfrog(ctx!.device);
      gpu.init(gpuState);
      const cpu = new Leapfrog();
      cpu.init(cpuState);

      for (let p = 1; p <= PERIODS; p++) {
        await stepMany(ctx!, gpu, gpuState, DT, STEPS_PER_PERIOD);
        for (let s = 0; s < STEPS_PER_PERIOD; s++) cpu.step(cpuState, DT);
        await gpu.readState(gpuState);
        let dev = 0;
        for (let i = 0; i < 3; i++) {
          dev = Math.max(
            dev,
            Math.hypot(gpuState.px[i] - cpuState.px[i], gpuState.py[i] - cpuState.py[i]),
          );
        }
        deviations.push(dev);
      }
      console.log(
        `figure-eight f32 deviation by period: ` +
          deviations.map((d, k) => `${k + 1}:${d.toExponential(1)}`).join(' '),
      );
    });

    it('stays on the choreography for the first periods', () => {
      // Measured: 1.3e-5 after one period, 4.3e-4 after ten. ~5× headroom.
      expect(deviations[0]).toBeLessThan(1e-4);
      expect(deviations[9]).toBeLessThan(2e-3);
    });

    it('holds the choreography through 40 periods (measured finding)', () => {
      // MEASURED FINDING, opposite of the naive expectation: the eight is
      // NOT lost. Deviation grows slowly and roughly linearly (no chaotic
      // blowup on this horizon) to 2.4e-3 after 40 periods — ~0.1% of the
      // ~2-unit system. Pinned at 10× that; if this fails, the Float32
      // behaviour changed and the README story must be re-measured.
      expect(deviations[PERIODS - 1]).toBeLessThan(2.5e-2);
    });
  });

  /**
   * Sun + planets: hierarchical, strongly stable — the opposite regime.
   * Float32 noise perturbs the orbits but nothing amplifies it. Measured:
   * per-planet radius stays within 1.0e-4 of the CPU-f64 radius over 3
   * outer-planet orbits (~26k steps), energy drift 7.3e-6. Pinned at ~10×
   * headroom.
   */
  describe('sun + planets (stable Kepler orbits)', () => {
    const DT = sunPlanets.dt; // 0.004
    const OUTER_PERIOD = 2 * Math.PI * Math.sqrt(3.1 ** 3); // a=3.1, GM≈1
    const CHECKPOINTS = 12;
    const STEPS_PER_CHECK = Math.round((3 * OUTER_PERIOD) / DT / CHECKPOINTS);

    let maxRadiusErr = 0;
    let gpuDrift = Number.NaN;

    beforeAll(async () => {
      const gpuState = sunPlanets.generate(0, 0);
      const cpuState = sunPlanets.generate(0, 0);
      const gpu = new GpuLeapfrog(ctx!.device);
      gpu.init(gpuState);
      const cpu = new Leapfrog();
      cpu.init(cpuState);
      const e0 = totalEnergy(gpuState);

      const radius = (s: SimState, i: number): number =>
        Math.hypot(s.px[i] - s.px[0], s.py[i] - s.py[0]);

      for (let c = 0; c < CHECKPOINTS; c++) {
        await stepMany(ctx!, gpu, gpuState, DT, STEPS_PER_CHECK);
        for (let s = 0; s < STEPS_PER_CHECK; s++) cpu.step(cpuState, DT);
        await gpu.readState(gpuState);
        for (let i = 1; i < gpuState.n; i++) {
          const err = Math.abs(radius(gpuState, i) - radius(cpuState, i)) / radius(cpuState, i);
          maxRadiusErr = Math.max(maxRadiusErr, err);
        }
      }
      gpuDrift = relativeEnergyDrift(totalEnergy(gpuState), e0);
      console.log(
        `sun+planets over 3 outer orbits: max radius deviation vs f64 = ` +
          `${maxRadiusErr.toExponential(2)}, gpu energy drift = ${gpuDrift.toExponential(2)}`,
      );
    });

    it('keeps every planet within a small fraction of its f64 radius', () => {
      expect(maxRadiusErr).toBeLessThan(1e-3);
    });

    it('keeps energy drift bounded at the Float32 scale', () => {
      expect(gpuDrift).toBeLessThan(5e-5);
    });
  });
});
