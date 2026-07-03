import { computeAccelerations } from './forces';
import type { Integrator } from './integrator';
import type { SimState } from './state';

/**
 * Leapfrog (velocity Verlet) in kick–drift–kick (KDK) form:
 *
 *   kick    v ← v + a(x)·dt/2
 *   drift   x ← x + v·dt
 *   kick    v ← v + a(x)·dt/2     (a recomputed at the new x)
 *
 * Why leapfrog and not Euler: leapfrog is symplectic and time-reversible.
 * Its energy error oscillates in a bounded band instead of growing
 * secularly, so orbits stay orbits over arbitrarily many steps. The test
 * suite demonstrates the contrast against Euler directly.
 *
 * Acceleration reuse: the closing kick of step n and the opening kick of
 * step n+1 both evaluate a at the same positions x_{n+1}. Accelerations
 * live in SimState and stay valid between steps (Integrator contract), so
 * each step performs exactly ONE O(N²) force evaluation — the naive
 * formulation would do two.
 *
 * Caveat: the symplectic energy behaviour assumes a FIXED dt. Changing dt
 * mid-run (the UI exposes a live timestep slider) breaks that structure
 * and produces a small one-off energy shift — expected, not a bug.
 */
export class Leapfrog implements Integrator {
  readonly name = 'leapfrog';

  init(state: SimState): void {
    computeAccelerations(state);
  }

  step(state: SimState, dt: number): void {
    const { n, px, py, vx, vy, ax, ay } = state;
    const h = 0.5 * dt;
    // Opening kick (accelerations carried over from the previous step) and
    // drift, fused into one pass.
    for (let i = 0; i < n; i++) {
      vx[i] += h * ax[i];
      vy[i] += h * ay[i];
      px[i] += dt * vx[i];
      py[i] += dt * vy[i];
    }
    // The single force evaluation of this step.
    computeAccelerations(state);
    // Closing kick; these accelerations are reused by the next step.
    for (let i = 0; i < n; i++) {
      vx[i] += h * ax[i];
      vy[i] += h * ay[i];
    }
    state.time += dt;
  }
}
