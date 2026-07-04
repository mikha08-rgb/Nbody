# N-Body Simulator — project conventions

Portfolio project; the code and the git history are meant to be read by
humans evaluating the work. Keep physics code readable and commented.
TypeScript strict mode, no `any`.

## Non-negotiable conventions

- **2D in every phase.** Phase 2 is a Barnes–Hut *quadtree*, not an octree.
- **Units: G = 1.** Scenarios choose mass/length scales; time follows.
- **Data layout: structure-of-arrays** — flat `Float64Array`s in `SimState`
  (`src/physics/state.ts`), never arrays of particle objects. Phase 3 GPU
  buffers will be Float32; the energy-drift diagnostic is the tool that
  will measure what that precision drop costs.
- **Softening ε is per-scenario** (`SimState.eps`), never a constant in the
  force kernel. ε = 0 for figure-eight and sun+planets (exact solutions of
  the unsoftened problem); ε > 0 for the disk galaxy.
- **The potential-energy diagnostic uses the same softened kernel as the
  force** (`src/physics/forces.ts`). Mixing kernels fakes energy drift.
- **One force evaluation per step** (brute O(N²) or Barnes–Hut, behind the
  `ForceCalculator` seam). Accelerations live in `SimState` so leapfrog's
  closing kick is reused as the next opening kick. Keep this invariant
  when adding integrators or force calculators.
- **Momentum is machine-precision on the brute path ONLY.** Barnes–Hut
  node–particle forces are not pairwise-symmetric, so Newton's third law
  holds only approximately and total momentum drifts slowly — expected and
  correct for the algorithm, do not "fix" it. The 1e-12 momentum test pins
  the brute path; the BH test asserts drift is merely small and bounded.
- **Brute force stays.** It is the correctness reference, the test oracle,
  and the default for small N — never delete or modify it to serve BH.
- **Fixed physics dt, decoupled from rAF** via the accumulator in
  `src/physics/simulation.ts`; substep cap drops excess time (no spiral).
- **Every scenario ends generation in the center-of-momentum frame**
  (`toCenterOfMomentumFrame`).
- **Seeded PRNG only** (`src/scenarios/rng.ts`); no `Math.random` anywhere.
- A cold disk growing clumps/spiral arms is real instability physics —
  do not add damping or "fixes" for it.

## Interface seams (extend here, don't restructure)

- `Integrator` (`src/physics/integrator.ts`) — leapfrog is the real one;
  Euler exists only as the tests' negative control.
- `ForceCalculator` (`src/physics/forces.ts`) — a `(state) => void` that
  writes `ax/ay`. Implementations: brute `computeAccelerations` and
  `BarnesHut.accelerations`; integrators take one at construction.
- Scenario registry (`src/scenarios/registry.ts`) — one file + one entry
  per new scenario. `addDisk` (`src/scenarios/disk-galaxy.ts`) is the
  reusable disk builder for multi-galaxy scenarios.

## Barnes–Hut quadtree (`src/physics/barnes-hut.ts`)

- **Flat node arena**: SoA typed arrays indexed by node id, rebuilt from
  scratch every step, grown geometrically, reused forever — no node
  objects, no closures in the hot path, ~zero garbage per step (the bench
  verifies this with a heap-spread check).
- **Child-slot encoding**: −1 empty · ≥ 0 node index · ≤ −2 particle chain
  headed by particle −(v+2), linked through a per-particle `next` array.
  There are no leaf node records.
- **Opening criterion**: squared θ test, side² < θ²·d², against distance
  to the node's center of mass. θ = 0 can never satisfy the strict
  inequality, so it degenerates to exact brute force — a tested contract
  (1e-12), not an accident. Monopole terms use the same softened kernel
  (`state.eps`) as the brute path.
- **Depth cap 32**: coincident/near-coincident particles chain into one
  bucket at the cap instead of splitting forever.
- Traversal is an iterative DFS over a pre-allocated stack (bound
  3·MAX_DEPTH+1); one tree walk per particle.

## Benchmark methodology

`npm run bench`: brute vs BH at θ = 0.5 (the pinned benchmark setting —
never tune θ up, skip particles, or cap traversal to improve numbers) on
the seeded disk, N ∈ {1k…50k}. Brute is probed with one step and skipped
— never extrapolated — past a 2 s cutoff. The max-N-at-60fps claim is the
largest benchmarked N with a BH step under 16.7 ms. Scenario defaults
that cite performance (galaxy-collision defaultN) derive from this bench.

## Test policy

Tolerances in `src/physics/*.test.ts` are spec contracts and the setups
are pinned (scenario, seed, N, dt, steps). **Never loosen a tolerance or
reshuffle a seed to make a test pass** — either the physics regressed
(investigate) or a deliberate scenario change made the pinned dt too
coarse (reduce dt and say why in the commit).

## Commands

`npm run dev` · `npm test` · `npm run bench` (headless, tsx) ·
`npm run build` (tsc + vite). Benchmark results and the max-N-at-60fps
claim in README.md are derived from `npm run bench` output — update them
by re-running, not by editing numbers.

## Phase roadmap

1. **Done:** brute-force CPU baseline, leapfrog, tests, diagnostics, bench.
2. **Done:** Barnes–Hut quadtree (θ opening angle; bench = speed, drift =
   accuracy), galaxy-collision scenario. Deliberately out of scope, at
   most future work: incremental tree updates, higher multipoles,
   adaptive/individual timesteps, workers/SIMD/wasm.
3. WebGPU compute port (Float32 buffers; drift quantifies precision loss).
