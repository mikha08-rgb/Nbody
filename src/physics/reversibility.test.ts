import { describe, expect, it } from 'vitest';
import { figureEight } from '../scenarios/figure-eight';
import { Euler } from './euler';
import type { Integrator } from './integrator';
import { Leapfrog } from './leapfrog';
import type { SimState } from './state';

/**
 * Time-reversibility: run forward, negate all velocities, run the same
 * number of steps again — a reversible integrator retraces its own
 * trajectory back to the start.
 *
 * Leapfrog is algebraically time-reversible, so the only residual is
 * accumulated floating-point error: positions must return within 1e-6
 * RELATIVE TO THE SYSTEM SIZE (not machine precision — rounding error
 * accumulates even though the algebra is exact). Euler is not reversible
 * and must miss by orders of magnitude.
 *
 * PINNED setup (figure-eight, ε = 0, dt = 0.005, 500 steps each way);
 * tolerances are spec contracts — do not loosen.
 */
const DT = 0.005;
const STEPS = 500;

/** System size scale: the largest pairwise separation at t = 0. */
function sizeScale(s: SimState): number {
  let scale = 0;
  for (let i = 0; i < s.n; i++) {
    for (let j = i + 1; j < s.n; j++) {
      scale = Math.max(scale, Math.hypot(s.px[j] - s.px[i], s.py[j] - s.py[i]));
    }
  }
  return scale;
}

/** Forward STEPS, negate velocities, back STEPS; return max position error / size scale. */
function reversalError(integrator: Integrator): number {
  const state = figureEight.generate(3, 0);
  const x0 = state.px.slice();
  const y0 = state.py.slice();
  const scale = sizeScale(state);

  integrator.init(state);
  for (let i = 0; i < STEPS; i++) {
    integrator.step(state, DT);
  }
  for (let i = 0; i < state.n; i++) {
    state.vx[i] = -state.vx[i];
    state.vy[i] = -state.vy[i];
  }
  for (let i = 0; i < STEPS; i++) {
    integrator.step(state, DT);
  }

  let maxErr = 0;
  for (let i = 0; i < state.n; i++) {
    maxErr = Math.max(maxErr, Math.hypot(state.px[i] - x0[i], state.py[i] - y0[i]));
  }
  return maxErr / scale;
}

describe('time-reversibility (pinned figure-eight trajectory)', () => {
  it('leapfrog returns to the start within 1e-6 of the system size', () => {
    expect(reversalError(new Leapfrog())).toBeLessThan(1e-6);
  });

  it('euler misses the start by orders of magnitude more', () => {
    const leapfrog = reversalError(new Leapfrog());
    const euler = reversalError(new Euler());
    expect(euler).toBeGreaterThan(100 * leapfrog);
  });
});
