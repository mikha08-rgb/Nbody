import type { SimState } from '../physics/state';

/**
 * A scenario is a pure generator: (n, seed) → fresh SimState.
 *
 * Adding a scenario is one file exporting one of these plus a line in
 * registry.ts; the UI picks it up automatically.
 */
export interface Scenario {
  readonly id: string;
  readonly label: string;
  /** Suggested integrator timestep (sim-time units per physics tick). */
  readonly dt: number;
  /** Body count used when the scenario is first selected. */
  readonly defaultN: number;
  /**
   * False → the body count is intrinsic to the scenario (e.g. exactly 3
   * for the figure-eight) and the UI's particle-count slider is disabled.
   */
  readonly supportsN: boolean;
  /**
   * Generate a fresh state. Must end by calling toCenterOfMomentumFrame so
   * the system doesn't drift off-screen and momentum starts at ~0.
   * Deterministic for a given (n, seed).
   */
  generate(n: number, seed: number): SimState;
}
