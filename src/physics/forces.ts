import type { SimState } from './state';

/**
 * Gravitational constant, in natural units.
 *
 * Working with G = 1 means a scenario chooses two of its three scales
 * (mass, length, time) freely and the third follows. The figure-eight, for
 * example, uses unit masses at ~unit separation, which makes its period
 * T ≈ 6.33 time units. Nothing in the physics core ever needs a
 * dimensionful G.
 */
export const G = 1;

/**
 * Brute-force pairwise gravitational accelerations, written into
 * state.ax/ay.
 *
 * Plummer-softened kernel:
 *
 *   a_i = Σ_j  G m_j (r_j − r_i) / (|r_j − r_i|² + ε²)^(3/2)
 *
 * ε > 0 bounds the force as two bodies approach, avoiding the 1/r²
 * singularity that would otherwise fling close pairs apart at any finite
 * timestep. ε comes from the scenario (state.eps) — deliberately not a
 * constant here, because scenarios that are exact solutions of the
 * unsoftened problem require ε = 0.
 *
 * Each unordered pair is visited once (j > i) and the interaction is
 * applied to both bodies with opposite signs — Newton's third law. That
 * halves the work and conserves total momentum to floating-point rounding.
 *
 * O(N²): this is the Phase 1 baseline that the Barnes–Hut quadtree
 * replaces in Phase 2.
 */
export function computeAccelerations(s: SimState): void {
  const { n, px, py, ax, ay, mass } = s;
  const eps2 = s.eps * s.eps;
  ax.fill(0);
  ay.fill(0);
  for (let i = 0; i < n; i++) {
    const xi = px[i];
    const yi = py[i];
    const mi = mass[i];
    // Accumulate body i's acceleration in locals so the inner loop touches
    // ax/ay only for body j.
    let axi = 0;
    let ayi = 0;
    for (let j = i + 1; j < n; j++) {
      const dx = px[j] - xi;
      const dy = py[j] - yi;
      const r2 = dx * dx + dy * dy + eps2;
      // (r² + ε²)^(−3/2) via one sqrt and one division.
      const invR3 = 1 / (r2 * Math.sqrt(r2));
      const sx = dx * invR3;
      const sy = dy * invR3;
      axi += G * mass[j] * sx;
      ayi += G * mass[j] * sy;
      ax[j] -= G * mi * sx;
      ay[j] -= G * mi * sy;
    }
    ax[i] += axi;
    ay[i] += ayi;
  }
}

/**
 * Total gravitational potential energy, using the SAME softened kernel as
 * the force calculation:
 *
 *   U = − Σ_{i<j}  G m_i m_j / √(|r_j − r_i|² + ε²)
 *
 * The softened acceleration above is exactly −∇ of this potential, so
 * KE + U_softened is the conserved quantity of the system we actually
 * integrate. Diagnosing drift against the bare Newtonian −G m m / r would
 * mix kernels and report spurious "drift" that no integrator can fix.
 *
 * O(N²) like the force kernel — call it on a diagnostics interval, not
 * every frame.
 */
export function potentialEnergy(s: SimState): number {
  const { n, px, py, mass } = s;
  const eps2 = s.eps * s.eps;
  let u = 0;
  for (let i = 0; i < n; i++) {
    const xi = px[i];
    const yi = py[i];
    const mi = mass[i];
    for (let j = i + 1; j < n; j++) {
      const dx = px[j] - xi;
      const dy = py[j] - yi;
      u -= (G * mi * mass[j]) / Math.sqrt(dx * dx + dy * dy + eps2);
    }
  }
  return u;
}
