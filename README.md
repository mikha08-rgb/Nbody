# N-Body Gravity Simulator

A 2D gravitational N-body simulator that runs in the browser — vanilla
TypeScript + Canvas 2D, no framework. This is **Phase 1: a correct,
readable CPU baseline**: brute-force O(N²) gravity, a symplectic leapfrog
integrator, and diagnostics that prove conservation laws hold, measured
rather than assumed.

Scenarios include a rotating disk galaxy (exponential profile, rotation
curve from the enclosed mass), the Chenciner–Montgomery three-body
figure-eight choreography, and a small planetary system.

## Run it

```sh
npm install
npm run dev      # dev server
npm test         # physics test suite (Vitest)
npm run bench    # headless benchmark (no rendering)
npm run build    # type-check + production build
```

Controls: drag to pan, scroll to zoom; overlay has play/pause, reset,
scenario selector, body count (regenerates the scenario), and a live
timestep slider.

## Physics

- **Leapfrog (kick–drift–kick)** integrator: symplectic and
  time-reversible, so energy error stays bounded instead of growing. The
  closing kick of one step reuses its force evaluation as the next step's
  opening kick — exactly one O(N²) evaluation per step.
- **Plummer softening** with per-scenario ε (ε = 0 for scenarios that are
  exact solutions of the unsoftened problem). The potential-energy
  diagnostic uses the same softened kernel as the force, so KE + U is the
  genuinely conserved quantity.
- **Structure-of-arrays** state in flat `Float64Array`s — cache-friendly
  now, and the layout Phase 3's GPU buffers will mirror.
- Fixed physics timestep decoupled from the render loop via an
  accumulator, with a substep cap that drops excess time rather than
  freezing a lagging tab.

The test suite pins concrete tolerances: orbital radius stable to 1% over
100 orbits, leapfrog energy drift < 10⁻⁴ over 1,000 steps (with Euler as a
failing control), momentum conserved to 10⁻¹² relative, and
time-reversibility to 10⁻⁶ of the system size.

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

- **Phase 1 (this)** — brute-force CPU baseline: correctness, tests,
  diagnostics, benchmark.
- **Phase 2 — Barnes–Hut quadtree**: O(N log N) force approximation;
  the benchmark above quantifies the speedup, the energy-drift diagnostic
  quantifies the accuracy cost of the opening-angle parameter θ.
- **Phase 3 — WebGPU compute**: port the force kernel to GPU compute
  shaders (Float32); the energy-drift diagnostic measures what the drop
  from Float64 costs.
