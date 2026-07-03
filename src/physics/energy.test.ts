import { describe, expect, it } from 'vitest';
import { getScenario } from '../scenarios/registry';
import { totalEnergy, relativeEnergyDrift } from './diagnostics';
import { Euler } from './euler';
import type { Integrator } from './integrator';
import { Leapfrog } from './leapfrog';

/**
 * Energy conservation contract.
 *
 * PINNED SETUP — scenario, seed, N, dt, and step count are deliberately
 * fixed so this test measures one reproducible trajectory. If it fails:
 * do NOT loosen the tolerance and do NOT reshuffle the seed. Either the
 * physics regressed (investigate), or a conscious scenario change made the
 * pinned dt too coarse (reduce DT here and say why in the commit).
 */
const SCENARIO = 'disk-galaxy';
const SEED = 42;
const N = 200;
const DT = 0.002;
const STEPS = 1000;

function runDrift(integrator: Integrator): number {
  const state = getScenario(SCENARIO).generate(N, SEED);
  integrator.init(state);
  const e0 = totalEnergy(state);
  for (let i = 0; i < STEPS; i++) {
    integrator.step(state, DT);
  }
  return relativeEnergyDrift(totalEnergy(state), e0);
}

describe('energy conservation (pinned disk-galaxy trajectory)', () => {
  it('leapfrog drifts less than 1e-4 over 1000 steps', () => {
    expect(runDrift(new Leapfrog())).toBeLessThan(1e-4);
  });

  it('euler drifts at least 100x more than leapfrog on the identical setup', () => {
    const leapfrog = runDrift(new Leapfrog());
    const euler = runDrift(new Euler());
    expect(euler).toBeGreaterThan(100 * leapfrog);
  });
});
