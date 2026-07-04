import type { SimState } from './state';

/**
 * Barnes–Hut quadtree (Phase 2) — the O(N log N) force approximation that
 * replaces the brute-force kernel for large N, behind the same
 * ForceCalculator seam (forces.ts).
 *
 * ## Node arena
 *
 * The tree is rebuilt from scratch every step, so the make-or-break for JS
 * performance is allocation discipline: no node objects, ever. Nodes live
 * in flat typed arrays (structure-of-arrays, like SimState itself), indexed
 * by integer id, reused across steps and grown geometrically when a step
 * needs more. Steady state allocates nothing.
 *
 * Only internal nodes exist in the arena — there are no leaf node records.
 * Particles hang directly off their parent's child slots via an encoding:
 *
 *   slot value v        meaning
 *   ------------        -------
 *   v === EMPTY (-1)    empty quadrant
 *   v >= 0              internal node with arena index v
 *   v <= -2             particle chain headed by particle index -(v + 2)
 *
 * Chains are singly linked through `next` (one Int32 per particle). Below
 * the depth cap every chain has length 1; at the cap, particles too close
 * to separate (or exactly coincident) pile into one bucket chain instead of
 * splitting forever — that bucket is the depth-cap safety valve.
 *
 * ## Build
 *
 * Root bounds are recomputed from the particle extent each step (a square,
 * so quadrants stay square). Insertion is iterative — descend by quadrant
 * comparison until an empty slot, an occupied slot to split, or the depth
 * cap. Mass and center of mass are then aggregated in a single bottom-up
 * pass: children are always created after their parent, so reverse arena
 * order visits every child before its parent.
 */

/**
 * Maximum tree depth. 2^-32 of the root side is far below Float64
 * resolution of any sane scenario scale, so a chain bucket at this depth
 * only ever holds effectively-coincident particles.
 */
const MAX_DEPTH = 32;

const EMPTY = -1;

/** Child-slot encoding helpers (see the table above). */
const encodeParticle = (i: number): number => -(i + 2);
const decodeParticle = (v: number): number => -v - 2;

export class BarnesHut {
  /**
   * Opening angle θ: a node of side s at distance d from the target
   * particle is used as a monopole iff s/d < θ, otherwise opened. Runtime
   * parameter (UI slider); θ = 0 can never satisfy the test, so every node
   * opens and the sum degenerates to exact brute force — a tested
   * guarantee, not an accident.
   */
  theta: number;

  // --- node arena (structure-of-arrays; grown geometrically, never shrunk)
  private capacity = 0;
  /** Nodes in use by the current tree. */
  private nodeCount = 0;
  private centerX = new Float64Array(0);
  private centerY = new Float64Array(0);
  /** Half side length of the node's square region. */
  private half = new Float64Array(0);
  private mass = new Float64Array(0);
  private comX = new Float64Array(0);
  private comY = new Float64Array(0);
  /** 4 child slots per node, encoded as in the table above. */
  private children = new Int32Array(0);

  /** Particle chain links (length ≥ n), EMPTY-terminated. */
  private next = new Int32Array(0);

  constructor(theta = 0.5) {
    this.theta = theta;
  }

  /**
   * Rebuild the tree for the current positions. Public for the invariant
   * tests; force evaluation calls it internally.
   */
  build(s: SimState): void {
    const { n, px, py } = s;
    this.ensureParticleCapacity(n);
    this.nodeCount = 0;
    if (n === 0) return;

    // Root bounds: the tightest square around the particle extent. A zero
    // extent (single particle, or all coincident) is legal — every insert
    // then rides the same-quadrant path down to the depth-cap bucket.
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    for (let i = 0; i < n; i++) {
      if (px[i] < minX) minX = px[i];
      if (px[i] > maxX) maxX = px[i];
      if (py[i] < minY) minY = py[i];
      if (py[i] > maxY) maxY = py[i];
    }
    const root = this.newNode(
      0.5 * (minX + maxX),
      0.5 * (minY + maxY),
      0.5 * Math.max(maxX - minX, maxY - minY),
    );

    for (let i = 0; i < n; i++) this.insert(s, root, i);
    this.aggregate(s);
  }

  /**
   * Insert particle i, starting at the root. Iterative: quadrant descent
   * with three exits — empty slot (place the particle), node slot (keep
   * descending), particle slot (split it into a child node, or chain at
   * the depth cap).
   */
  private insert(s: SimState, root: number, i: number): void {
    const { px, py } = s;
    const x = px[i];
    const y = py[i];
    let node = root;
    let depth = 0;
    for (;;) {
      // Quadrant bit layout: bit 0 = east (x ≥ center), bit 1 = north.
      const q = (x >= this.centerX[node] ? 1 : 0) | (y >= this.centerY[node] ? 2 : 0);
      const slot = node * 4 + q;
      const v = this.children[slot];

      if (v === EMPTY) {
        this.next[i] = EMPTY;
        this.children[slot] = encodeParticle(i);
        return;
      }
      if (v >= 0) {
        node = v;
        depth++;
        continue;
      }
      // Slot holds a particle. At the cap, chain instead of splitting —
      // this is what makes coincident particles terminate.
      if (depth + 1 >= MAX_DEPTH) {
        this.next[i] = decodeParticle(v);
        this.children[slot] = encodeParticle(i);
        return;
      }
      // Split: push a child node into the slot, move the resident particle
      // down into it (below the cap chains have length 1), keep descending.
      const child = this.newChild(node, q);
      this.children[slot] = child;
      const r = decodeParticle(v);
      const rq =
        (px[r] >= this.centerX[child] ? 1 : 0) | (py[r] >= this.centerY[child] ? 2 : 0);
      this.children[child * 4 + rq] = v;
      node = child;
      depth++;
    }
  }

  /**
   * Bottom-up mass / center-of-mass pass (monopole only — no higher
   * multipoles in this phase). newNode always appends, so children have
   * strictly larger indices than their parent and a reverse scan sees
   * every child's aggregate before its parent needs it.
   */
  private aggregate(s: SimState): void {
    const { px, py, mass } = s;
    for (let node = this.nodeCount - 1; node >= 0; node--) {
      let m = 0;
      let mx = 0;
      let my = 0;
      for (let slot = node * 4; slot < node * 4 + 4; slot++) {
        const v = this.children[slot];
        if (v === EMPTY) continue;
        if (v >= 0) {
          m += this.mass[v];
          mx += this.mass[v] * this.comX[v];
          my += this.mass[v] * this.comY[v];
        } else {
          for (let p = decodeParticle(v); p !== EMPTY; p = this.next[p]) {
            m += mass[p];
            mx += mass[p] * px[p];
            my += mass[p] * py[p];
          }
        }
      }
      this.mass[node] = m;
      // m = 0 can only happen for massless particles; park the COM at the
      // node center so it stays finite.
      this.comX[node] = m > 0 ? mx / m : this.centerX[node];
      this.comY[node] = m > 0 ? my / m : this.centerY[node];
    }
  }

  private newNode(cx: number, cy: number, half: number): number {
    if (this.nodeCount === this.capacity) this.grow();
    const id = this.nodeCount++;
    this.centerX[id] = cx;
    this.centerY[id] = cy;
    this.half[id] = half;
    this.children.fill(EMPTY, id * 4, id * 4 + 4);
    return id;
  }

  private newChild(parent: number, q: number): number {
    const h = 0.5 * this.half[parent];
    return this.newNode(
      this.centerX[parent] + (q & 1 ? h : -h),
      this.centerY[parent] + (q & 2 ? h : -h),
      h,
    );
  }

  /** Double the node arena, copying live nodes. Rare and amortized. */
  private grow(): void {
    const capacity = Math.max(256, this.capacity * 2);
    const copy = <T extends Float64Array | Int32Array>(old: T, next: T): T => {
      next.set(old);
      return next;
    };
    this.centerX = copy(this.centerX, new Float64Array(capacity));
    this.centerY = copy(this.centerY, new Float64Array(capacity));
    this.half = copy(this.half, new Float64Array(capacity));
    this.mass = copy(this.mass, new Float64Array(capacity));
    this.comX = copy(this.comX, new Float64Array(capacity));
    this.comY = copy(this.comY, new Float64Array(capacity));
    this.children = copy(this.children, new Int32Array(capacity * 4));
    this.capacity = capacity;
  }

  private ensureParticleCapacity(n: number): void {
    if (this.next.length < n) {
      this.next = new Int32Array(Math.max(n, this.next.length * 2));
    }
  }

  // --- inspection (tests / diagnostics; allocation here is fine) ---------

  /** Number of arena nodes in the current tree. */
  get nodes(): number {
    return this.nodeCount;
  }

  /** Total mass aggregated at the root. */
  get rootMass(): number {
    return this.nodeCount > 0 ? this.mass[0] : 0;
  }

  /** Root center of mass. */
  get rootCom(): { x: number; y: number } {
    return this.nodeCount > 0
      ? { x: this.comX[0], y: this.comY[0] }
      : { x: 0, y: 0 };
  }

  /**
   * How many leaf chains reference each particle. The tree invariant is
   * that every entry is exactly 1 — each particle lives in exactly one
   * leaf slot.
   */
  leafReferenceCounts(n: number): Int32Array {
    const counts = new Int32Array(n);
    for (let slot = 0; slot < this.nodeCount * 4; slot++) {
      const v = this.children[slot];
      if (v <= -2) {
        for (let p = decodeParticle(v); p !== EMPTY; p = this.next[p]) counts[p]++;
      }
    }
    return counts;
  }
}
