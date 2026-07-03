import { G } from '../physics/forces';
import { createState, toCenterOfMomentumFrame, type SimState } from '../physics/state';
import { mulberry32, type Rng } from './rng';
import type { Scenario } from './types';

/**
 * A cold, rotationally supported disk around a dominant central mass — the
 * Phase 1 stand-in for a galaxy, and the scenario the benchmark and the
 * Phase 2 Barnes–Hut comparison run on.
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
const R_MAX = 4; // outer truncation

/** Draw a radius from pdf(r) ∝ r·e^(−r/R_d) on [R_MIN, R_MAX] (peak at r = R_d). */
function sampleRadius(rng: Rng): number {
  const peak = R_D * Math.exp(-1);
  for (;;) {
    const r = R_MIN + (R_MAX - R_MIN) * rng();
    if (rng() * peak <= r * Math.exp(-r / R_D)) return r;
  }
}

function generate(n: number, seed: number): SimState {
  const rng = mulberry32(seed);

  // Softening scaled to the mean interparticle spacing (uniform-area
  // estimate), so neighbours can't slingshot each other at any N.
  const spacing = Math.sqrt((Math.PI * R_MAX * R_MAX) / n);
  const eps = 0.5 * spacing;

  // Body 0 is the central mass; n disk particles follow.
  const s = createState(n + 1, eps);
  s.mass[0] = CENTRAL_MASS;

  const particleMass = DISK_MASS / n;
  for (let b = 1; b <= n; b++) {
    const r = sampleRadius(rng);
    const theta = 2 * Math.PI * rng();
    s.px[b] = r * Math.cos(theta);
    s.py[b] = r * Math.sin(theta);
    s.mass[b] = particleMass;
  }

  // Circular velocities from the enclosed mass: walk particles in radius
  // order, accumulating the mass interior to each.
  const order = Array.from({ length: n }, (_, i) => i + 1);
  order.sort((a, b) => Math.hypot(s.px[a], s.py[a]) - Math.hypot(s.px[b], s.py[b]));
  let enclosed = CENTRAL_MASS;
  for (const b of order) {
    const r = Math.hypot(s.px[b], s.py[b]);
    const v = Math.sqrt((G * enclosed) / r);
    // Tangential, counterclockwise: v * (-y, x) / r.
    s.vx[b] = (-v * s.py[b]) / r;
    s.vy[b] = (v * s.px[b]) / r;
    enclosed += particleMass;
  }

  toCenterOfMomentumFrame(s);
  return s;
}

export const diskGalaxy: Scenario = {
  id: 'disk-galaxy',
  label: 'Disk galaxy',
  dt: 0.004,
  defaultN: 1500,
  supportsN: true,
  generate,
};
