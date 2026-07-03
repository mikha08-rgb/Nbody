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
- **One O(N²) force evaluation per step.** Accelerations live in `SimState`
  so leapfrog's closing kick is reused as the next opening kick. Keep this
  invariant when adding integrators or force calculators.
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
- Force kernel (`src/physics/forces.ts`) — Phase 2's Barnes–Hut replaces
  `computeAccelerations` behind the same shape.
- Scenario registry (`src/scenarios/registry.ts`) — one file + one entry
  per new scenario.

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
2. Barnes–Hut quadtree (θ opening angle; bench = speed, drift = accuracy).
3. WebGPU compute port (Float32 buffers; drift quantifies precision loss).
