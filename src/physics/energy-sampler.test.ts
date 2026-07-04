import { describe, expect, it } from 'vitest';
import { getScenario } from '../scenarios/registry';
import { totalEnergy } from './diagnostics';
import { EnergySampler } from './energy-sampler';
import { Leapfrog } from './leapfrog';

/**
 * The sampler's contract is EXACTNESS: same Float64 arithmetic, same
 * softened kernel, same accumulation order as diagnostics.totalEnergy —
 * only the scheduling differs. So the tests assert bit-identical results
 * (toBe, not toBeCloseTo): any reordering of the sums is a contract change,
 * not a rounding nuisance.
 */
describe('time-sliced energy sampler', () => {
  it('reproduces totalEnergy bit-for-bit in a single unbounded tick', () => {
    const state = getScenario('disk-galaxy').generate(500, 42);
    const sampler = new EnergySampler();
    sampler.begin(state);
    expect(sampler.sampling).toBe(true);
    const result = sampler.tick(Infinity);
    expect(result).toBe(totalEnergy(state));
    expect(sampler.sampling).toBe(false);
  });

  it('is isolated from concurrent stepping and independent of slice size', () => {
    const state = getScenario('disk-galaxy').generate(400, 7);
    const expected = totalEnergy(state);

    const sampler = new EnergySampler();
    sampler.begin(state);

    // Step the live system while the sample is in progress; a zero budget
    // forces the smallest possible slices (one clock-check block per tick).
    const integrator = new Leapfrog();
    integrator.init(state);
    let result: number | null = null;
    let ticks = 0;
    while (result === null) {
      integrator.step(state, 0.004);
      result = sampler.tick(0);
      ticks++;
    }

    expect(ticks).toBeGreaterThan(1); // it really was sliced
    expect(result).toBe(expected); // snapshot semantics: begin()-time state
    expect(totalEnergy(state)).not.toBe(expected); // the live state moved on
  });

  it('can be re-begun mid-sample (scenario reset while sampling)', () => {
    const big = getScenario('disk-galaxy').generate(400, 1);
    const small = getScenario('disk-galaxy').generate(100, 2);
    const sampler = new EnergySampler();
    sampler.begin(big);
    expect(sampler.tick(0)).toBeNull(); // leave a sample in progress
    sampler.begin(small);
    expect(sampler.tick(Infinity)).toBe(totalEnergy(small));
  });
});
