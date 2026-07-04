# N-Body Gravity Simulator

A 2D gravitational N-body simulator that runs in the browser — vanilla
TypeScript + Canvas 2D, no framework. Phase 1 built **a correct, readable
CPU baseline**: brute-force O(N²) gravity, a symplectic leapfrog
integrator, and diagnostics that prove conservation laws hold, measured
rather than assumed. Phase 2 adds a **Barnes–Hut quadtree** — O(N log N)
forces verified against that baseline — lifting the 60 fps ceiling from
4,000 to 10,000 bodies and making a 10k–50k particle galaxy collision the
payoff demo.

Scenarios include a rotating disk galaxy (exponential profile, rotation
curve from the enclosed mass), a two-galaxy collision on a bound orbit
(tidal tails, counter-rotating secondary), the Chenciner–Montgomery
three-body figure-eight choreography, and a small planetary system.

## Run it

```sh
npm install
npm run dev      # dev server
npm test         # physics test suite (Vitest)
npm run bench    # headless benchmark (no rendering)
npm run build    # type-check + production build
```

Controls: drag to pan, scroll to zoom; overlay has play/pause, reset,
scenario selector, body count (regenerates the scenario), a live timestep
slider, a force-method selector (brute force / Barnes–Hut), and a live θ
slider for the Barnes–Hut opening angle.

## Physics

- **Leapfrog (kick–drift–kick)** integrator: symplectic and
  time-reversible, so energy error stays bounded instead of growing. The
  closing kick of one step reuses its force evaluation as the next step's
  opening kick — exactly one O(N²) evaluation per step.
- **Plummer softening** with per-scenario ε (ε = 0 for scenarios that are
  exact solutions of the unsoftened problem). The potential-energy
  diagnostic uses the same softened kernel as the force, so KE + U is the
  genuinely conserved quantity.
- **Barnes–Hut quadtree** (Phase 2): far-away groups of bodies act as
  single point masses (monopole at the center of mass), chosen by the
  standard opening test s/d < θ. **What θ trades off:** larger θ accepts
  coarser far-field approximations — faster steps, larger force error
  (RMS ~0.4% at the default θ = 0.5 on the benchmark disk); θ = 0 opens
  every node and degenerates to exact brute force, a tested guarantee.
  The tree is rebuilt every step into a flat pre-allocated typed-array
  arena — no node objects, near-zero garbage per step (measured, see the
  benchmark). The brute-force kernel stays as the correctness reference
  and the default for small N.
- **An honest caveat, by design:** Barnes–Hut forces are not
  pairwise-symmetric, so Newton's third law — and with it exact momentum
  conservation — does not survive the approximation. Momentum drifts
  slowly (~2.5×10⁻⁴ relative over the pinned 1,000-step test) instead of
  holding at 10⁻¹²; that is inherent to the algorithm, and the energy
  overlay doubles as a live accuracy readout for θ.
- **Structure-of-arrays** state in flat `Float64Array`s — cache-friendly
  now, and the layout Phase 3's GPU buffers will mirror.
- Fixed physics timestep decoupled from the render loop via an
  accumulator, with a substep cap that drops excess time rather than
  freezing a lagging tab.

The test suite pins concrete tolerances: orbital radius stable to 1% over
100 orbits, leapfrog energy drift < 10⁻⁴ over 1,000 steps (with Euler as a
failing control), momentum conserved to 10⁻¹² relative (brute-force path),
and time-reversibility to 10⁻⁶ of the system size. Phase 2 adds: quadtree
root mass/COM match direct summation, every particle in exactly one leaf,
θ = 0 equal to brute force within 10⁻¹², RMS force error < 1% at θ = 0.5
and monotonically improving as θ shrinks, bounded energy and momentum
drift under leapfrog+BH, and termination on degenerate (coincident)
inputs.

## Benchmark

`npm run bench` times the physics step (no rendering) on the seeded disk
galaxy, brute force vs Barnes–Hut at θ = 0.5 (the benchmark setting —
θ is never tuned up to flatter the numbers). On an Apple Silicon laptop
(Node 25, single thread):

| Bodies | Brute ms/step | BH (θ=0.5) ms/step | Speedup |
|-------:|--------------:|--------------------:|--------:|
| 1,000 | 0.83 | 0.81 | 1.0× |
| 2,000 | 3.09 | 2.10 | 1.5× |
| 5,000 | 20.20 | 7.01 | 2.9× |
| 10,000 | 81.47 | 16.41 | 5.0× |
| 20,000 | 326.33 | 38.11 | 8.6× |
| 50,000 | — (> 2 s, skipped) | 107.64 | — |

- **Crossover:** Barnes–Hut matches brute force at 1,000 bodies (within
  noise) and wins decisively from 2,000 up — tree overhead stops paying
  for itself only below ~1k, which is why brute force stays the default
  for small scenarios.
- **Max bodies at 60 fps: 10,000 with Barnes–Hut** (16.41 ms/step — at
  the edge of the 16.7 ms budget, so with rendering on top expect
  slightly under 60). The Phase 1 brute-force ceiling was 4,000.
- Brute force at 50,000 is skipped, not extrapolated: a single probe
  step exceeded the 2 s cutoff. The measured trend (×4 per doubling) is
  textbook O(N²); Barnes–Hut grows at roughly O(N log N), ~2.2–2.8× per
  doubling in this range.
- **Allocation discipline:** the bench samples heap usage across 300
  Barnes–Hut steps at N = 20,000 — spread 0.02 MB, i.e. the per-step
  tree rebuild allocates essentially nothing (flat pre-allocated node
  arena, no node objects).

## Roadmap

- **Phase 1 (done)** — brute-force CPU baseline: correctness, tests,
  diagnostics, benchmark.
- **Phase 2 (done)** — Barnes–Hut quadtree: O(N log N) forces verified
  against the brute-force oracle; the benchmark above quantifies the
  speedup (crossover ~1–2k bodies, 8.6× at 20k), the energy-drift
  diagnostic quantifies the accuracy cost of θ. Deliberately not built:
  incremental tree updates, higher-order multipoles, adaptive timesteps.
- **Phase 3 — WebGPU compute**: port the force kernel to GPU compute
  shaders (Float32); the energy-drift diagnostic measures what the drop
  from Float64 costs. Profiling note from Phase 2: the per-particle tree
  traversal is embarrassingly parallel and dominates the step, so workers
  or SIMD would help long before the GPU port — both deliberately left
  out of scope until Phase 3 decides the compute story wholesale.
