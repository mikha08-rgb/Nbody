import { G } from '../physics/forces';
import { createState, toCenterOfMomentumFrame, type SimState } from '../physics/state';
import { mulberry32 } from './rng';
import type { Scenario } from './types';

/**
 * A star with a handful of light planets on circular orbits — the "does it
 * look like a solar system" sanity scenario.
 *
 * ε = 0: planet masses are small enough that there are no violent close
 * encounters, and Kepler orbits are exact solutions of the unsoftened
 * problem that softening would distort.
 *
 * Circular speed uses the mass enclosed by each orbit (star + interior
 * planets) — the same principle as the disk galaxy. At these mass ratios
 * it differs from √(G·M★/r) by well under a percent, but consistency is
 * free. Planet-planet tugs remain as genuine perturbations.
 */
const STAR_MASS = 1;

const PLANETS = [
  { r: 0.5, m: 2e-4 },
  { r: 0.8, m: 6e-4 },
  { r: 1.3, m: 1.2e-3 },
  { r: 2.0, m: 8e-4 },
  { r: 3.1, m: 2.5e-3 },
];

function generate(_n: number, seed: number): SimState {
  const rng = mulberry32(seed);
  const s = createState(1 + PLANETS.length, 0);
  s.mass[0] = STAR_MASS;
  let enclosed = STAR_MASS;
  for (let i = 0; i < PLANETS.length; i++) {
    const { r, m } = PLANETS[i];
    // Orbits are circular, so the starting angle is pure variety — seeded
    // for reproducibility like everything else.
    const theta = 2 * Math.PI * rng();
    const v = Math.sqrt((G * enclosed) / r);
    const b = i + 1;
    s.px[b] = r * Math.cos(theta);
    s.py[b] = r * Math.sin(theta);
    // Tangential, counterclockwise: (-sin θ, cos θ).
    s.vx[b] = -v * Math.sin(theta);
    s.vy[b] = v * Math.cos(theta);
    s.mass[b] = m;
    enclosed += m;
  }
  toCenterOfMomentumFrame(s);
  return s;
}

export const sunPlanets: Scenario = {
  id: 'sun-planets',
  label: 'Sun + planets',
  dt: 0.004,
  defaultN: 1 + PLANETS.length,
  supportsN: false,
  generate,
};
