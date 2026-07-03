/**
 * Simulation state in structure-of-arrays (SoA) layout.
 *
 * All simulation data lives in flat typed arrays rather than an array of
 * particle objects: the hot O(N²) force loop reads memory sequentially
 * (cache-friendly, no pointer chasing), and the layout maps 1:1 onto the
 * GPU storage buffers planned for Phase 3.
 *
 * Precision note: CPU arrays are Float64. Phase 3 GPU buffers will be
 * Float32; the energy-drift diagnostic built in this phase is the yardstick
 * that will quantify exactly what that loss of precision costs.
 *
 * Units: G = 1 "natural" units (see forces.ts). Each scenario picks its own
 * mass/length scale; the time unit then follows from G = 1.
 */
export interface SimState {
  /** Number of bodies. */
  readonly n: number;
  px: Float64Array;
  py: Float64Array;
  vx: Float64Array;
  vy: Float64Array;
  /**
   * Accelerations are part of the state so integrators can reuse the closing
   * force evaluation of step n as the opening one of step n+1 — exactly one
   * O(N²) evaluation per step (see leapfrog.ts and the Integrator contract).
   * Invariant between steps: ax/ay match the current px/py.
   */
  ax: Float64Array;
  ay: Float64Array;
  mass: Float64Array;
  /**
   * Plummer softening length ε (see forces.ts). A per-scenario parameter,
   * not a global constant: exact-solution scenarios (figure-eight,
   * sun+planets) need ε = 0, while the disk galaxy needs ε > 0 to tame
   * close encounters.
   */
  readonly eps: number;
  /** Elapsed simulation time. */
  time: number;
}

export function createState(n: number, eps: number): SimState {
  return {
    n,
    px: new Float64Array(n),
    py: new Float64Array(n),
    vx: new Float64Array(n),
    vy: new Float64Array(n),
    ax: new Float64Array(n),
    ay: new Float64Array(n),
    mass: new Float64Array(n),
    eps,
    time: 0,
  };
}

/**
 * Shift a freshly generated scenario into the center-of-momentum frame:
 * center of mass at the origin, total momentum ≈ 0.
 *
 * Every scenario generator must call this last. Without it, any net
 * momentum from generation makes the whole system drift off-screen, and
 * the momentum-conservation diagnostic would start from a nonzero baseline
 * instead of ~machine epsilon.
 */
export function toCenterOfMomentumFrame(s: SimState): void {
  const { n, px, py, vx, vy, mass } = s;
  let m = 0;
  let comX = 0;
  let comY = 0;
  let pX = 0;
  let pY = 0;
  for (let i = 0; i < n; i++) {
    m += mass[i];
    comX += mass[i] * px[i];
    comY += mass[i] * py[i];
    pX += mass[i] * vx[i];
    pY += mass[i] * vy[i];
  }
  comX /= m;
  comY /= m;
  const ux = pX / m;
  const uy = pY / m;
  for (let i = 0; i < n; i++) {
    px[i] -= comX;
    py[i] -= comY;
    vx[i] -= ux;
    vy[i] -= uy;
  }
}
