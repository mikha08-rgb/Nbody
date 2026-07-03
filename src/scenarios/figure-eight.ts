import { createState, toCenterOfMomentumFrame, type SimState } from '../physics/state';
import type { Scenario } from './types';

/**
 * The Chenciner–Montgomery figure-eight: three equal masses chasing each
 * other along a single ∞-shaped curve. A periodic exact solution of the
 * UNSOFTENED three-body problem (Chenciner & Montgomery 2000, with the
 * initial conditions found numerically by Simó).
 *
 * The numbers below are the published values, hardcoded on purpose — they
 * are not derivable in closed form. ε must be 0: softening changes the
 * potential, and the choreography slowly deforms under it.
 *
 * With G = 1 and unit masses the period is T ≈ 6.32591398. This scenario
 * doubles as a correctness beacon: if the integrator is wrong, the eight
 * visibly falls apart within a few periods.
 */
const X1 = 0.97000436;
const Y1 = -0.24308753;
const VX3 = -0.93240737;
const VY3 = -0.86473146;

function generate(_n: number, _seed: number): SimState {
  const s = createState(3, 0); // ε = 0 — exact solution of the unsoftened problem
  s.mass.fill(1);
  // Bodies 1 and 2 start at mirrored positions with half the (negated)
  // velocity of body 3, which starts at the origin.
  s.px[0] = X1;
  s.py[0] = Y1;
  s.px[1] = -X1;
  s.py[1] = -Y1;
  s.px[2] = 0;
  s.py[2] = 0;
  s.vx[0] = -VX3 / 2;
  s.vy[0] = -VY3 / 2;
  s.vx[1] = -VX3 / 2;
  s.vy[1] = -VY3 / 2;
  s.vx[2] = VX3;
  s.vy[2] = VY3;
  // Already a center-of-momentum configuration by construction; the shift
  // is a no-op up to rounding but keeps the scenario contract uniform.
  toCenterOfMomentumFrame(s);
  return s;
}

export const figureEight: Scenario = {
  id: 'figure-eight',
  label: 'Figure-eight (3-body)',
  dt: 0.005,
  defaultN: 3,
  supportsN: false,
  generate,
};
