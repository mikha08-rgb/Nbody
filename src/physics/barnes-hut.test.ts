import { describe, expect, it } from 'vitest';
import { getScenario } from '../scenarios/registry';
import { BarnesHut } from './barnes-hut';
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
  });

  it('handles an empty state', () => {
    const bh = new BarnesHut();
    bh.build(createState(0, 0));
    expect(bh.nodes).toBe(0);
    expect(bh.rootMass).toBe(0);
  });
});
