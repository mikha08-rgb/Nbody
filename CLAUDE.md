# N-Body Simulator — project conventions

Portfolio project; the code and the git history are meant to be read by
humans evaluating the work. Keep physics code readable and commented.
TypeScript strict mode, no `any`.

## Non-negotiable conventions

- **2D in every phase.** Phase 2 is a Barnes–Hut *quadtree*, not an octree.
- **Units: G = 1.** Scenarios choose mass/length scales; time follows.
- **Data layout: structure-of-arrays** — flat `Float64Array`s in `SimState`
  (`src/physics/state.ts`), never arrays of particle objects. The Phase 3
  GPU buffers mirror it 1:1 as one `array<f32>` per state array; the
  measured cost of that precision drop lives in README's precision story
  and the pinned `*.gpu.test.ts` bounds.
- **Softening ε is per-scenario** (`SimState.eps`), never a constant in the
  force kernel. ε = 0 for figure-eight and sun+planets (exact solutions of
  the unsoftened problem); ε > 0 for the disk galaxy.
- **The potential-energy diagnostic uses the same softened kernel as the
  force** (`src/physics/forces.ts`). Mixing kernels fakes energy drift.
- **One force evaluation per step** (brute O(N²), Barnes–Hut, or the GPU
  force pass). Accelerations live in `SimState` (or its GPU buffers) so
  leapfrog's closing kick is reused as the next opening kick. Keep this
  invariant when adding integrators or force calculators.
- **Momentum is machine-precision on the CPU brute path ONLY.** Barnes–Hut
  node–particle forces are not pairwise-symmetric, so Newton's third law
  holds only approximately and total momentum drifts slowly — expected and
  correct for the algorithm, do not "fix" it. The 1e-12 momentum test pins
  the CPU brute path; the BH test asserts drift is merely small and
  bounded. The GPU brute path is pairwise-symmetric only in exact
  arithmetic — each body sums independently in f32, so its momentum test
  pins a measured rounding-scale bound (~5e-8), not 1e-12.
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
  Euler exists only as the tests' negative control. Phase 3 adds
  `GpuLeapfrog` (whole KDK step on the GPU) and `SwitchableIntegrator`
  (CPU ↔ GPU delegation) behind the same interface; `Simulation` was not
  touched.
- `ForceCalculator` (`src/physics/forces.ts`) — a `(state) => void` that
  writes `ax/ay`. Implementations: brute `computeAccelerations` and
  `BarnesHut.accelerations`; integrators take one at construction. The
  GPU path is NOT a ForceCalculator — that seam is synchronous and
  WebGPU readback is not; the GPU lands behind `Integrator` instead (see
  the WebGPU section).
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

## WebGPU port (`src/gpu/`, Phase 3)

- **Seam decision:** the GPU is a new `Integrator` (`GpuLeapfrog`), not a
  `ForceCalculator`. Reason: only readback is async in WebGPU — encoding
  and submitting are synchronous host calls — so a GPU-resident KDK step
  fits `Integrator.step()` as-is, while a GPU ForceCalculator would need
  a per-step await and force the whole pipeline async. Do not "fix" the
  seam; this was the architecture decision of the phase.
- **State ownership:** while GPU mode is active, the Float32 GPU buffers
  are authoritative and `SimState`'s Float64 arrays are a read-back
  MIRROR, refreshed once per rendered frame through a 2-deep snapshot
  ring (`NBodyGpu.requestSnapshot`); renderer and diagnostics read the
  mirror, which lags the physics by ≤2 frames. Ring saturation is the
  backpressure signal: the frame loop skips `advance()` and sim time
  slows to GPU pace (drop-excess, like the substep cap — never queue
  unbounded dispatches).
- **Buffers:** one `array<f32>` storage buffer per SimState array
  (posX/posY/velX/velY/accX/accY/mass) + a 16-byte uniform
  (n, eps², dt, dt/2). `NBodyPrograms` (module/layout/pipelines) lives
  for the session; `NBodyGpu` (buffers, fixed n) is rebuilt per scenario
  (re)generation. WGSL lives in `nbody-kernels.wgsl` (imported `?raw`);
  `WORKGROUP_SIZE = 256` must match between .wgsl and .ts.
- **Entering GPU mode quantizes the live state to Float32 in place**
  (positions/velocities/masses, written back so mirror == GPU exactly),
  and E₀ re-baselines from the quantized state. This is the documented
  one-off cost of the port; don't hide it and don't double-count it as
  drift. `state.eps` stays Float64; the shader gets f32(ε²).
- **Same physics, f32:** the WGSL force pass uses the identical softened
  kernel expression as `forces.ts` (`1.0 / (r2 * sqrt(r2))`) — no
  `inverseSqrt`, no fast-math reshuffling, no Kahan/compensated tricks to
  flatter drift. Self-interaction is skipped by index (0/0 = NaN at
  ε = 0); the partial last tile is bounded, not padded with fake bodies.
- **Measured precision findings** (Apple metal-3; re-measure, don't
  assume, if the kernel changes): pinned-disk energy drift is
  INDISTINGUISHABLE from Float64 through 10k steps (dt² truncation
  dominates f32 rounding); momentum drifts at ~5e-8 of scale (third-law
  cancellation only to rounding); the figure-eight choreography survives
  40+ periods (deviation 2.4e-3, roughly linear growth). Details and
  exact numbers: README + `src/gpu/*.gpu.test.ts`.
- **The energy diagnostic survives GPU-scale N by time-slicing, never by
  approximation**: `EnergySampler` (`src/physics/energy-sampler.ts`)
  spreads the exact Float64 O(N²) sum across frames (~3 ms slices) from
  a snapshot; results are bit-identical to `totalEnergy` (tested).
  Sample cadence adapts to measured cost. Don't cap the diagnostic N and
  don't compute energy in f32 on the GPU.
- **Feature detection:** `acquireGpu()` (`src/gpu/device.ts`) is the only
  entry point; null → the UI disables the GPU option with a visible note
  and Phases 1–2 behavior is exactly preserved. Device loss falls back
  to the CPU from the mirror and retires the option.

## Benchmark methodology

`npm run bench`: brute vs BH at θ = 0.5 (the pinned benchmark setting —
never tune θ up, skip particles, or cap traversal to improve numbers) on
the seeded disk, N ∈ {1k…50k}. Brute is probed with one step and skipped
— never extrapolated — past a 2 s cutoff. The max-N-at-60fps claim is the
largest benchmarked N with a BH step under 16.7 ms. Scenario defaults
that cite performance (galaxy-collision defaultN) derive from this bench.

`npm run bench:gpu` (Node can't time WebGPU): drives `/bench.html`
headlessly via Playwright — GPU f32 brute vs CPU BH θ = 0.5 in the same
page on the same machine, N ∈ {2k…500k}, same probe/skip rule for both,
best-of-3 reps, GPU timed submit-to-onSubmittedWorkDone (wall-clock
throughput, not encode time). The adapter label is part of the output;
never publish GPU numbers without the hardware next to them. The 60 fps
claims are physics-only — Canvas 2D is the frame ceiling at GPU-scale N
(documented in README; renderer rewrite is future work, not this phase).

## Test policy

Tolerances in `src/physics/*.test.ts` are spec contracts and the setups
are pinned (scenario, seed, N, dt, steps). **Never loosen a tolerance or
reshuffle a seed to make a test pass** — either the physics regressed
(investigate) or a deliberate scenario change made the pinned dt too
coarse (reduce dt and say why in the commit).

GPU tests (`src/gpu/*.gpu.test.ts`, `npm run test:gpu`) run in real
Chromium via Vitest browser mode — Node has no WebGPU, and Playwright's
headless *shell* doesn't either; `vitest.gpu.config.ts` selects the full
build (`channel: 'chromium'`). No adapter → suites skip with a loud
warning, never silently green. GPU float math is not bit-reproducible
across vendors, so GPU bounds are MEASURED-then-pinned with ~5–10×
headroom (each test comments its measured value and hardware) instead of
pinned bit-exact trajectories. The same no-loosening rule applies; if a
bound moves, the shader physics changed — re-measure and update the
README precision story in the same commit.

## Commands

`npm run dev` (sim at `/`, GPU bench at `/bench.html`) · `npm test`
(CPU, Node) · `npm run test:gpu` (browser mode, needs WebGPU) ·
`npm run bench` (headless CPU, tsx) · `npm run bench:gpu` (headless
browser GPU) · `npm run build` (tsc + vite). Benchmark results and the
max-N-at-60fps claims in README.md derive from `npm run bench` /
`npm run bench:gpu` output — update them by re-running, not by editing
numbers.

## Phase roadmap

1. **Done:** brute-force CPU baseline, leapfrog, tests, diagnostics, bench.
2. **Done:** Barnes–Hut quadtree (θ opening angle; bench = speed, drift =
   accuracy), galaxy-collision scenario. Deliberately out of scope, at
   most future work: incremental tree updates, higher multipoles,
   adaptive/individual timesteps, workers/SIMD/wasm.
3. **Done:** WebGPU compute port — GPU-resident Float32 leapfrog behind
   the Integrator seam, browser GPU bench, measured precision story.
   Deliberately out of scope, at most future work: GPU tree build (the
   past-500k answer), WebGL/WebGPU point-sprite renderer (the measured
   frame-rate ceiling at GPU-scale N), workers/SIMD on the CPU path.
