import { G } from '../physics/forces';
import { createState, toCenterOfMomentumFrame, type SimState } from '../physics/state';
import { addDisk, diskSoftening } from './disk-galaxy';
import { mulberry32 } from './rng';
import type { Scenario } from './types';

/**
 * Two disk galaxies on an approach trajectory — the Phase 2 payoff demo,
 * sized for Barnes-Hut (tens of thousands of bodies; brute force at the
 * default N would run at seconds per frame).
 *
 * Built from two addDisk calls (disk-galaxy.ts) with bulk offsets and
 * velocities; each disk rotates in equilibrium with its own mass only, so
 * everything that happens once they overlap — tidal tails, the bridge,
 * the eventual merger — is genuine encounter dynamics.
 */

/** Secondary/primary mass ratio (applies to central body and disk alike). */
const MASS_RATIO = 0.5;

/** Initial separation of the two centers, in disk radii terms (R_MAX = 4). */
const SEPARATION = 10;

/**
 * Perpendicular offset between the approach lines. A head-on merger is
 * visually dull; the impact parameter gives the pair orbital angular
 * momentum, which is what stretches out tidal tails.
 */
const IMPACT_PARAMETER = 2.5;

/**
 * Relative approach speed as a fraction of the parabolic (zero-energy)
 * speed at the initial separation. < 1 means the pair is bound: they make
 * a close passage, fling out tails, fall back and merge.
 */
const PARABOLIC_FRACTION = 0.7;

/** Total mass of one galaxy (central body + disk) at unit scale. */
const GALAXY_MASS = 1 + 0.35;

/**
 * The secondary counter-rotates (spin −1). Prograde vs retrograde
 * encounters respond very differently to tides — the retrograde disk
 * holds together longer — and the asymmetry reads well on screen.
 */
function generate(n: number, seed: number): SimState {
  const rng = mulberry32(seed);

  // Split the particle budget so both disks get the same per-particle
  // mass: the secondary carries MASS_RATIO of the primary's disk mass and
  // therefore MASS_RATIO of its particles.
  const n1 = Math.round(n / (1 + MASS_RATIO));
  const n2 = n - n1;

  const m1 = GALAXY_MASS;
  const m2 = GALAXY_MASS * MASS_RATIO;
  const vParabolic = Math.sqrt((2 * G * (m1 + m2)) / SEPARATION);
  const vRel = PARABOLIC_FRACTION * vParabolic;

  // Positions and velocities split inversely by mass, so the pair's center
  // of momentum starts near the origin (toCenterOfMomentumFrame zeroes the
  // remainder). Approach along x, offset along y.
  const f1 = m2 / (m1 + m2);
  const f2 = m1 / (m1 + m2);

  const s = createState(n + 2, diskSoftening(n1));
  const after = addDisk(
    s,
    0,
    {
      centralMass: 1,
      diskMass: 0.35,
      n: n1,
      x: -SEPARATION * f1,
      y: -IMPACT_PARAMETER * f1,
      vx: vRel * f1,
      vy: 0,
      spin: 1,
    },
    rng,
  );
  addDisk(
    s,
    after,
    {
      centralMass: MASS_RATIO,
      diskMass: 0.35 * MASS_RATIO,
      n: n2,
      x: SEPARATION * f2,
      y: IMPACT_PARAMETER * f2,
      vx: -vRel * f2,
      vy: 0,
      spin: -1,
    },
    rng,
  );

  toCenterOfMomentumFrame(s);
  return s;
}

export const galaxyCollision: Scenario = {
  id: 'galaxy-collision',
  label: 'Galaxy collision',
  dt: 0.004,
  defaultN: 20000,
  maxN: 50000,
  supportsN: true,
  forceMethod: 'barnes-hut',
  generate,
};
