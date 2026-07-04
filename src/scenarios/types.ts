import type { SimState } from '../physics/state';

/**
 * Which force calculator drives the simulation. Brute force is exact and
 * wins at small N; Barnes-Hut approximates (opening angle θ) and wins at
 * large N — the benchmark table in README.md pins the crossover.
 */
export type ForceMethod = 'brute' | 'barnes-hut';

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
   * Upper bound for the UI body-count slider. Scenarios sized for
   * Barnes-Hut raise this well past what brute force sustains.
   * Defaults to 4,000 when omitted.
   */
  readonly maxN?: number;
  /**
   * Force method selected when this scenario is chosen (the user can still
   * switch afterwards). Defaults to 'brute' when omitted.
   */
  readonly forceMethod?: ForceMethod;
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
