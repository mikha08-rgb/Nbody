import { G } from '../physics/forces';
import { createState, toCenterOfMomentumFrame, type SimState } from '../physics/state';
import { mulberry32, type Rng } from './rng';
import type { Scenario } from './types';

/**
 * A cold, rotationally supported disk around a dominant central mass — the
 * Phase 1 stand-in for a galaxy, the benchmark scenario, and (via addDisk)
 * the building block of the Phase 2 galaxy-collision scenario.
 *
 * Surface density: exponential, Σ(r) ∝ e^(−r/R_d) — the standard profile
 * for galactic disks — sampled by rejection.
 *
 * Rotation: the circular speed at radius r must balance the gravity of ALL
 * mass enclosed within r — central body plus the disk mass interior to the
 * particle — not just the central body:
 *
 *     v_circ(r) = √( G · M_enc(r) / r )
 *
 * (Enclosed disk mass is the sum over particles with r_j < r_i; treating
 * it as if it were centrally concentrated is exact for spherical shells
 * and a good approximation for a thin disk at this fidelity.) Using only
 * the central mass would leave the outer disk under-supported by
 * ~√(1 + M_disk/M_c) and it would visibly reshuffle within the first
 * orbit.
 *
 * Expect structure anyway: a perfectly cold disk is gravitationally
 * unstable and grows clumps and spiral arms over a few rotations. That is
 * real disk-instability physics, not a bug — do NOT add damping or other
 * "fixes" for it.
 */
const CENTRAL_MASS = 1;
const DISK_MASS = 0.35;
const R_D = 1; // exponential scale length
const R_MIN = 0.2; // inner cutoff — keeps particles off the central body
export const DISK_R_MAX = 4; // outer truncation

/** Draw a radius from pdf(r) ∝ r·e^(−r/R_d) on [R_MIN, R_MAX] (peak at r = R_d). */
function sampleRadius(rng: Rng): number {
  const peak = R_D * Math.exp(-1);
  for (;;) {
    const r = R_MIN + (DISK_R_MAX - R_MIN) * rng();
    if (rng() * peak <= r * Math.exp(-r / R_D)) return r;
  }
}

/** One disk galaxy's bulk parameters, for composing multi-galaxy scenarios. */
export interface DiskSpec {
  centralMass: number;
  diskMass: number;
  /** Disk particles (the central body is extra). */
  n: number;
  /** Bulk position of the galaxy's center. */
  x: number;
  y: number;
  /** Bulk velocity, added on top of the internal rotation. */
  vx: number;
  vy: number;
  /** +1 counterclockwise, −1 clockwise. */
  spin: 1 | -1;
}

/**
 * Write one disk galaxy (central body + spec.n particles) into s starting
 * at body index `start`; returns the next free index. Rotation balances
 * only the disk's OWN enclosed mass — a companion galaxy's pull is part of
 * the encounter dynamics, not the equilibrium.
 */
export function addDisk(s: SimState, start: number, spec: DiskSpec, rng: Rng): number {
  s.px[start] = spec.x;
  s.py[start] = spec.y;
  s.vx[start] = spec.vx;
  s.vy[start] = spec.vy;
  s.mass[start] = spec.centralMass;

  const particleMass = spec.diskMass / spec.n;
  for (let b = start + 1; b <= start + spec.n; b++) {
    const r = sampleRadius(rng);
    const theta = 2 * Math.PI * rng();
    s.px[b] = spec.x + r * Math.cos(theta);
    s.py[b] = spec.y + r * Math.sin(theta);
    s.mass[b] = particleMass;
  }

  // Circular velocities from the enclosed mass: walk particles in radius
  // order, accumulating the mass interior to each.
  const order = Array.from({ length: spec.n }, (_, i) => start + 1 + i);
  const radius = (b: number): number => Math.hypot(s.px[b] - spec.x, s.py[b] - spec.y);
  order.sort((a, b) => radius(a) - radius(b));
  let enclosed = spec.centralMass;
  for (const b of order) {
    const r = radius(b);
    const v = (spec.spin * Math.sqrt((G * enclosed) / r)) / r;
    // Tangential: v_circ · (−y, x) / r, flipped by spin.
    s.vx[b] = spec.vx - v * (s.py[b] - spec.y);
    s.vy[b] = spec.vy + v * (s.px[b] - spec.x);
    enclosed += particleMass;
  }
  return start + spec.n + 1;
}

/**
 * Softening scaled to the mean interparticle spacing (uniform-area
 * estimate over the disk), so neighbours can't slingshot each other at
 * any N.
 */
export function diskSoftening(n: number): number {
  return 0.5 * Math.sqrt((Math.PI * DISK_R_MAX * DISK_R_MAX) / n);
}

function generate(n: number, seed: number): SimState {
  const rng = mulberry32(seed);
  const s = createState(n + 1, diskSoftening(n));
  addDisk(
    s,
    0,
    { centralMass: CENTRAL_MASS, diskMass: DISK_MASS, n, x: 0, y: 0, vx: 0, vy: 0, spin: 1 },
    rng,
  );
  toCenterOfMomentumFrame(s);
  return s;
}

export const diskGalaxy: Scenario = {
  id: 'disk-galaxy',
  label: 'Disk galaxy',
  dt: 0.004,
  defaultN: 1500,
  maxN: 20000,
  supportsN: true,
  generate,
};
