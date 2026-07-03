import { describe, expect, it } from 'vitest';
import { getScenario } from '../scenarios/registry';
import { momentumScale, totalMomentum } from './diagnostics';
import { Leapfrog } from './leapfrog';

/**
 * Momentum conservation from a center-of-momentum start.
 *
 * The pair-symmetric force kernel applies exactly opposite interactions to
 * each pair, so total momentum should be conserved to floating-point
 * rounding — not merely to integrator accuracy.
 *
 * Bound (spec contract): |p_total| < 1e-12 · Σ mᵢ|vᵢ|. The relative form
 * keeps the bound meaningful regardless of a scenario's mass/velocity
 * scale. PINNED setup; do not loosen.
 */
const SEED = 7;
const N = 300;
const DT = 0.004;
const STEPS = 500;

describe('momentum conservation (pinned disk-galaxy trajectory)', () => {
  it('keeps |p| below 1e-12 of the momentum scale after 500 steps', () => {
    const state = getScenario('disk-galaxy').generate(N, SEED);
    const integrator = new Leapfrog();
    integrator.init(state);
    for (let i = 0; i < STEPS; i++) {
      integrator.step(state, DT);
    }
    expect(totalMomentum(state).mag).toBeLessThan(1e-12 * momentumScale(state));
  });
});
