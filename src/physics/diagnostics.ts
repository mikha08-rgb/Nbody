import { potentialEnergy } from './forces';
import type { SimState } from './state';

/** Total kinetic energy, Σ ½ m v². */
export function kineticEnergy(s: SimState): number {
  const { n, vx, vy, mass } = s;
  let ke = 0;
  for (let i = 0; i < n; i++) {
    ke += 0.5 * mass[i] * (vx[i] * vx[i] + vy[i] * vy[i]);
  }
  return ke;
}

/**
 * Total energy: kinetic + SOFTENED potential (see forces.ts for why the
 * potential must use the same ε-softened kernel as the force).
 *
 * O(N²) because of the potential — compute on a diagnostics interval
 * (every ~0.5 s of wall time), never per frame.
 */
export function totalEnergy(s: SimState): number {
  return kineticEnergy(s) + potentialEnergy(s);
}

/** Total linear momentum and its magnitude. */
export function totalMomentum(s: SimState): { x: number; y: number; mag: number } {
  const { n, vx, vy, mass } = s;
  let pX = 0;
  let pY = 0;
  for (let i = 0; i < n; i++) {
    pX += mass[i] * vx[i];
    pY += mass[i] * vy[i];
  }
  return { x: pX, y: pY, mag: Math.hypot(pX, pY) };
}

/**
 * Scale for judging momentum conservation: Σ mᵢ|vᵢ|. The relative bound
 * |p_total| / Σ mᵢ|vᵢ| is scale-independent — it means the same thing
 * whether a scenario uses heavy fast bodies or light slow ones.
 */
export function momentumScale(s: SimState): number {
  const { n, vx, vy, mass } = s;
  let scale = 0;
  for (let i = 0; i < n; i++) {
    scale += mass[i] * Math.hypot(vx[i], vy[i]);
  }
  return scale;
}

/**
 * Relative energy drift |E − E₀| / |E₀|, where E₀ is the total energy at
 * scenario start. This is THE headline diagnostic of the project: it
 * validates the integrator now, and in Phase 3 it will measure what the
 * Float32 GPU port costs in precision.
 */
export function relativeEnergyDrift(e: number, e0: number): number {
  return Math.abs(e - e0) / Math.abs(e0);
}
