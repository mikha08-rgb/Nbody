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
galaxy. On an Apple Silicon laptop (Node 25, single thread):

| Bodies | ms / step | Steps / frame budget (16.7 ms) |
|-------:|----------:|-------------------------------:|
| 500 | 0.209 | 79.7 |
| 1,000 | 0.841 | 19.8 |
| 2,000 | 3.427 | 4.9 |
| 4,000 | 13.267 | 1.3 |

Derived from the table: **4,000 bodies is the largest benchmarked count
whose step time (13.27 ms) fits inside the 16.7 ms / 60 fps frame
budget** — with little room left for rendering at that size; 2,000 bodies
(3.43 ms/step) runs with ample headroom. Step time scales as N², as
expected for the brute-force kernel. This table is the before/after
baseline for Phase 2.

## Roadmap

- **Phase 1 (this)** — brute-force CPU baseline: correctness, tests,
  diagnostics, benchmark.
- **Phase 2 — Barnes–Hut quadtree**: O(N log N) force approximation;
  the benchmark above quantifies the speedup, the energy-drift diagnostic
  quantifies the accuracy cost of the opening-angle parameter θ.
- **Phase 3 — WebGPU compute**: port the force kernel to GPU compute
  shaders (Float32); the energy-drift diagnostic measures what the drop
  from Float64 costs.
