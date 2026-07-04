import { G } from './forces';
import type { SimState } from './state';

/**
 * Time-sliced exact total-energy sampling.
 *
 * The energy diagnostic is O(N²) in the potential term, and it MUST stay
 * exact — Float64, same softened kernel as forces.ts — because it is the
 * yardstick for integrator quality (Phase 1) and for what the Float32 GPU
 * port costs (Phase 3). But a single synchronous sample at GPU-scale N
 * would freeze the page for seconds to minutes. Rather than capping N or
 * approximating the potential, the sampler snapshots the state and spreads
 * the exact pair sum over many frames in millisecond-sized slices.
 *
 * The result is identical (bit-for-bit) to diagnostics.totalEnergy on the
 * snapshot: the kinetic term is taken whole at begin() (it is only O(N)),
 * and the potential accumulates row by row in the same i<j order as
 * forces.potentialEnergy.
 *
 * The intended cadence consequence: samples get sparser as N² grows — the
 * caller spaces samples by a multiple of the measured compute cost
 * (lastCostMs). Exact and honest at every N, never a multi-frame hitch.
 */
export class EnergySampler {
  // Snapshot of the state under measurement. Copies, not references: the
  // live simulation keeps stepping while the sum is in progress.
  private px = new Float64Array(0);
  private py = new Float64Array(0);
  private mass = new Float64Array(0);
  private n = 0;
  private eps2 = 0;

  /** Kinetic part, computed whole at begin(). */
  private kinetic = 0;
  /** Potential accumulated so far, over rows [0, row). */
  private potential = 0;
  private row = 0;
  private active = false;
  private computeMs = 0;

  /** Total compute time (ms) of the most recently completed sample. */
  lastCostMs = 0;

  /** True while a sample is in progress. */
  get sampling(): boolean {
    return this.active;
  }

  /**
   * Snapshot `state` and start a new sample (replacing any in progress).
   * Velocities are consumed immediately by the O(N) kinetic sum, so only
   * positions and masses need to live in the snapshot.
   */
  begin(state: SimState): void {
    const { n, px, py, vx, vy, mass } = state;
    if (this.px.length < n) {
      this.px = new Float64Array(n);
      this.py = new Float64Array(n);
      this.mass = new Float64Array(n);
    }
    this.px.set(px);
    this.py.set(py);
    this.mass.set(mass);
    this.n = n;
    this.eps2 = state.eps * state.eps;

    // Same accumulation order as diagnostics.kineticEnergy.
    let ke = 0;
    for (let i = 0; i < n; i++) {
      ke += 0.5 * mass[i] * (vx[i] * vx[i] + vy[i] * vy[i]);
    }
    this.kinetic = ke;
    this.potential = 0;
    this.row = 0;
    this.computeMs = 0;
    this.active = true;
  }

  /**
   * Work on the sample for roughly `budgetMs` of wall time. Returns the
   * total energy when the sample completes, null while still in progress.
   *
   * The clock is checked once per CHECK_PAIRS pair terms rather than per
   * row: at small N a row is a handful of operations and per-row clock
   * reads would dominate; at large N a single row can approach a
   * millisecond, which is exactly the slice granularity.
   */
  tick(budgetMs: number): number | null {
    if (!this.active) return null;
    const { px, py, mass, n, eps2 } = this;
    const t0 = performance.now();
    let pairsSinceCheck = 0;
    let u = this.potential;
    let i = this.row;
    while (i < n) {
      const xi = px[i];
      const yi = py[i];
      const mi = mass[i];
      for (let j = i + 1; j < n; j++) {
        const dx = px[j] - xi;
        const dy = py[j] - yi;
        u -= (G * mi * mass[j]) / Math.sqrt(dx * dx + dy * dy + eps2);
      }
      pairsSinceCheck += n - i - 1;
      i++;
      if (pairsSinceCheck >= EnergySampler.CHECK_PAIRS) {
        pairsSinceCheck = 0;
        if (performance.now() - t0 >= budgetMs) break;
      }
    }
    this.row = i;
    this.potential = u;
    this.computeMs += performance.now() - t0;
    if (i < n) return null;
    this.active = false;
    this.lastCostMs = this.computeMs;
    return this.kinetic + u;
  }

  /** Pair terms between clock checks (~a fraction of a ms of work). */
  private static readonly CHECK_PAIRS = 1 << 16;
}
