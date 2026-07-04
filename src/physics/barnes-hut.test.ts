import { describe, expect, it } from 'vitest';
import { getScenario } from '../scenarios/registry';
import { BarnesHut } from './barnes-hut';
import { momentumScale, relativeEnergyDrift, totalEnergy, totalMomentum } from './diagnostics';
import { computeAccelerations } from './forces';
import { Leapfrog } from './leapfrog';
import { createState } from './state';

/**
 * Barnes–Hut quadtree contracts.
 *
 * PINNED SETUP (scenario, seed, N) — same policy as the Phase 1 tests: a
 * failure means the tree regressed, not that the tolerance needs loosening.
 */
const SEED = 11;
const N = 2000;

function diskState(n = N) {
  return getScenario('disk-galaxy').generate(n, SEED);
}

describe('quadtree build invariants (pinned disk galaxy)', () => {
  it('aggregates the exact total mass and center of mass at the root', () => {
    const s = diskState();
    const bh = new BarnesHut();
    bh.build(s);

    let m = 0;
    let comX = 0;
    let comY = 0;
    for (let i = 0; i < s.n; i++) {
      m += s.mass[i];
      comX += s.mass[i] * s.px[i];
      comY += s.mass[i] * s.py[i];
    }
    comX /= m;
    comY /= m;

    // "Exact" here means the same floating-point sum up to summation
    // order: the tree accumulates per-quadrant subtotals, so the last few
    // ULPs may differ from the flat left-to-right sum. 1e-13 relative is
    // ~1000x looser than one ULP and far tighter than any physics effect.
    expect(Math.abs(bh.rootMass - m)).toBeLessThanOrEqual(1e-13 * m);
    const com = bh.rootCom;
    // COM tolerance is relative to the disk radius (the coordinate scale),
    // not to |com| itself — the COM sits at ~0 by construction.
    expect(Math.abs(com.x - comX)).toBeLessThanOrEqual(1e-13 * 4);
    expect(Math.abs(com.y - comY)).toBeLessThanOrEqual(1e-13 * 4);
  });

  it('places every particle in exactly one leaf', () => {
    const s = diskState();
    const bh = new BarnesHut();
    bh.build(s);
    const counts = bh.leafReferenceCounts(s.n);
    for (let i = 0; i < s.n; i++) {
      expect(counts[i]).toBe(1);
    }
  });
});

/**
 * RMS over particles of |a_bh − a_brute| / |a_brute|, on identical
 * positions. Disk accelerations are all O(0.1–10) here, so the
 * per-particle denominator is safe.
 */
function rmsAccelerationError(theta: number): number {
  const brute = diskState();
  computeAccelerations(brute);
  const bh = new BarnesHut(theta);
  const s = diskState();
  bh.accelerations(s);
  let sumSq = 0;
  for (let i = 0; i < s.n; i++) {
    const rel =
      Math.hypot(s.ax[i] - brute.ax[i], s.ay[i] - brute.ay[i]) /
      Math.hypot(brute.ax[i], brute.ay[i]);
    sumSq += rel * rel;
  }
  return Math.sqrt(sumSq / s.n);
}

describe('Barnes-Hut vs brute force (pinned disk galaxy)', () => {
  it('degenerates to brute force at theta = 0', () => {
    // theta = 0 opens every node by construction (s^2 < 0 never holds), so
    // only exact pairwise terms remain. The residual is pure summation
    // order: brute accumulates pair-symmetrically (j > i), the tree in
    // traversal order. Measured max ~7e-15; 1e-12 is the spec bound.
    const brute = diskState();
    computeAccelerations(brute);
    const s = diskState();
    new BarnesHut(0).accelerations(s);
    for (let i = 0; i < s.n; i++) {
      const rel =
        Math.hypot(s.ax[i] - brute.ax[i], s.ay[i] - brute.ay[i]) /
        Math.hypot(brute.ax[i], brute.ay[i]);
      expect(rel).toBeLessThan(1e-12);
    }
  });

  it('stays under 1% RMS error at theta = 0.5 and improves monotonically', () => {
    const errors = [0.8, 0.5, 0.3].map(rmsAccelerationError);
    expect(errors[1]).toBeLessThan(0.01); // theta = 0.5: measured ~3.9e-3
    expect(errors[0]).toBeGreaterThan(errors[1]); // 0.8 worse than 0.5
    expect(errors[1]).toBeGreaterThan(errors[2]); // 0.5 worse than 0.3
  });
});

describe('conservation under leapfrog + Barnes-Hut (pinned disk galaxy)', () => {
  // PINNED trajectory: N = 1000, seed 11, theta = 0.5, dt = 0.004,
  // 1000 steps. Same never-loosen policy as the Phase 1 contracts.
  const THETA = 0.5;
  const DT = 0.004;
  const STEPS = 1000;

  function run() {
    const s = diskState(1000);
    const integrator = new Leapfrog(new BarnesHut(THETA).accelerations);
    integrator.init(s);
    const e0 = totalEnergy(s);
    for (let i = 0; i < STEPS; i++) integrator.step(s, DT);
    return { s, e0 };
  }

  it('keeps energy drift bounded (< 1e-2 over 1000 steps)', () => {
    const { s, e0 } = run();
    // Measured ~1.7e-4: leapfrog's symplectic bound plus the O(theta)
    // monopole error. The 1e-2 spec bound is the Phase 2 contract; drift
    // shrinks as theta -> 0.
    expect(relativeEnergyDrift(totalEnergy(s), e0)).toBeLessThan(1e-2);
  });

  it('keeps momentum drift small and bounded (NOT machine precision)', () => {
    const { s } = run();
    // Why not 1e-12 like the brute-force test: Barnes-Hut node-particle
    // interactions are not pairwise-symmetric (i may see a monopole
    // containing j while j resolves i exactly), so Newton's third law -
    // and with it machine-precision momentum conservation - does not
    // survive the approximation. A slow drift is expected and correct.
    // Measured ~2.5e-4 on this pinned trajectory; the bound is ~4x that.
    expect(totalMomentum(s).mag / momentumScale(s)).toBeLessThan(1e-3);
  });
});

describe('quadtree degenerate inputs', () => {
  it('handles a single particle', () => {
    const s = createState(1, 0);
    s.px[0] = 3;
    s.py[0] = -2;
    s.mass[0] = 5;
    const bh = new BarnesHut();
    bh.build(s);
    expect(bh.rootMass).toBe(5);
    expect(bh.rootCom).toEqual({ x: 3, y: -2 });
    expect(bh.leafReferenceCounts(1)[0]).toBe(1);
  });

  it('terminates on all-coincident particles (depth-cap bucket)', () => {
    // 64 particles at the same point: zero extent, so quadrant descent can
    // never separate them. The build must bottom out at the depth cap and
    // chain them into one bucket rather than recursing forever.
    const n = 64;
    const s = createState(n, 0.01);
    for (let i = 0; i < n; i++) {
      s.px[i] = 1.5;
      s.py[i] = 1.5;
      s.mass[i] = 2;
    }
    const bh = new BarnesHut();
    bh.build(s);
    expect(bh.rootMass).toBe(n * 2);
    const counts = bh.leafReferenceCounts(n);
    for (let i = 0; i < n; i++) {
      expect(counts[i]).toBe(1);
    }
    // Forces on the coincident bucket must come out finite (zero, by
    // symmetry): every pair sits at distance 0 with eps > 0.
    bh.accelerations(s);
    for (let i = 0; i < n; i++) {
      expect(s.ax[i]).toBe(0);
      expect(s.ay[i]).toBe(0);
    }
  });

  it('handles an empty state', () => {
    const bh = new BarnesHut();
    bh.build(createState(0, 0));
    expect(bh.nodes).toBe(0);
    expect(bh.rootMass).toBe(0);
  });
});
