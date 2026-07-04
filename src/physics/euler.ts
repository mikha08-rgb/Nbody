import { computeAccelerations, type ForceCalculator } from './forces';
import type { Integrator } from './integrator';
import type { SimState } from './state';

/**
 * Explicit (forward) Euler:
 *
 *   x ← x + v(t)·dt
 *   v ← v + a(x(t))·dt
 *
 * Deliberately the textbook non-symplectic integrator. It exists as the
 * negative control for the test suite: same force kernel and step counts
 * as leapfrog, but visibly growing energy drift and no time-reversibility.
 * Not wired into the UI; do not use it for anything else.
 *
 * Order matters: positions must advance with the OLD velocities. Updating
 * v first would give semi-implicit (symplectic) Euler, which conserves
 * energy well and would spoil the comparison.
 */
export class Euler implements Integrator {
  readonly name = 'euler';

  /** Defaults to the brute-force kernel, same as Leapfrog. */
  constructor(private readonly computeForces: ForceCalculator = computeAccelerations) {}

  init(state: SimState): void {
    this.computeForces(state);
  }

  step(state: SimState, dt: number): void {
    const { n, px, py, vx, vy, ax, ay } = state;
    for (let i = 0; i < n; i++) {
      // x with old v, then v with a(x(t)) — ax/ay still match the
      // pre-drift positions (Integrator contract).
      px[i] += dt * vx[i];
      py[i] += dt * vy[i];
      vx[i] += dt * ax[i];
      vy[i] += dt * ay[i];
    }
    this.computeForces(state);
    state.time += dt;
  }
}
