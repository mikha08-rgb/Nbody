import type { SimState } from './state';

/**
 * Interface seam for time integrators (Phase 1 ships leapfrog and, for the
 * test suite's negative control, explicit Euler; later phases can slot in
 * higher-order or GPU-driven schemes without touching callers).
 *
 * Contract: after init() or any number of step() calls, state.ax/ay hold
 * accelerations consistent with the current positions. Integrators rely on
 * this to reuse the previous step's closing force evaluation instead of
 * recomputing it, keeping the cost at exactly one O(N²) evaluation per step.
 */
export interface Integrator {
  readonly name: string;
  /**
   * Prime state.ax/ay from the current positions. Call once after a
   * scenario is (re)generated, before the first step().
   */
  init(state: SimState): void;
  /** Advance the system by one fixed timestep dt. */
  step(state: SimState, dt: number): void;
}
