/**
 * mulberry32 — a tiny, fast, seedable PRNG.
 *
 * Every random scenario takes an explicit seed so runs are reproducible:
 * tests can pin a trajectory, the benchmark always times the same particle
 * distribution, and visual before/after comparisons see the same galaxy.
 * (Math.random is not seedable, hence this.)
 */
export type Rng = () => number;

export function mulberry32(seed: number): Rng {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
