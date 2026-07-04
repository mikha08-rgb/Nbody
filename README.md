# N-Body Gravity Simulator

A 2D gravitational N-body simulator that runs in the browser — vanilla
TypeScript + Canvas 2D, no framework. Phase 1 built **a correct, readable
CPU baseline**: brute-force O(N²) gravity, a symplectic leapfrog
integrator, and diagnostics that prove conservation laws hold, measured
rather than assumed. Phase 2 adds a **Barnes–Hut quadtree** — O(N log N)
forces verified against that baseline — lifting the 60 fps ceiling from
4,000 to 10,000 bodies and making a 10k–50k particle galaxy collision the
payoff demo. Phase 3 ports the brute-force kernel to **WebGPU compute
shaders in Float32**: the exact O(N²) sum beats CPU Barnes–Hut from
2,000 bodies up on an Apple M4 Pro and steps half a million bodies in
under two seconds — with the Float64 diagnostics measuring precisely
what the precision drop costs (measured answer: far less than expected;
see the precision story below).

Scenarios include a rotating disk galaxy (exponential profile, rotation
curve from the enclosed mass), a two-galaxy collision on a bound orbit
(tidal tails, counter-rotating secondary), the Chenciner–Montgomery
three-body figure-eight choreography, and a small planetary system.

## Run it

```sh
npm install
npx playwright install chromium   # one-time: browser for test:gpu / bench:gpu
npm run dev       # dev server (simulator at /, GPU benchmark at /bench.html)
npm test          # CPU physics test suite (Vitest, Node)
npm run test:gpu  # GPU test suite (Vitest browser mode, real Chromium+WebGPU)
npm run bench     # headless CPU benchmark (no rendering)
npm run bench:gpu # browser GPU benchmark, run headlessly via Playwright
npm run build     # type-check + production build
```

(Playwright has no postinstall hook, so the Chromium download really is a
separate one-time step; only `test:gpu` and `bench:gpu` need it.)

Controls: drag to pan, scroll to zoom; overlay has play/pause, reset,
scenario selector, body count (regenerates the scenario), a live timestep
slider, a force-method selector (brute force / Barnes–Hut / GPU), and a
live θ slider for the Barnes–Hut opening angle. The GPU option appears
disabled with a note in browsers without WebGPU; everything else works
identically there.

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
- **WebGPU compute** (Phase 3): the brute-force kernel as three WGSL
  compute passes (kick-drift, force, kick) over Float32 storage buffers —
  one `array<f32>` per state array, the same structure-of-arrays layout
  as the CPU. The whole KDK integration is GPU-resident: a step is a
  synchronous host-side submit, positions read back asynchronously once
  per rendered frame (the drawn picture lags the physics by a frame or
  two; if the GPU falls behind, the sim slows down rather than queueing
  unbounded work — same drop-excess philosophy as the substep cap). The
  force pass is the classic workgroup-tiled O(N²) reduction, same
  softened kernel expression as the CPU, no fast-math tricks. Entering
  GPU mode quantizes the system to Float32 once (documented, measured);
  diagnostics stay Float64 on the CPU, computed from read-back state.
- **Structure-of-arrays** state in flat `Float64Array`s — cache-friendly
  on the CPU, and mapped 1:1 onto the GPU buffers in Phase 3, exactly as
  planned in Phase 1.
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
inputs. Phase 3 adds a GPU suite (`npm run test:gpu`, Vitest browser mode
in real Chromium, since Node has no WebGPU): GPU accelerations match the
Float64 oracle within Float32-justified tolerances (measured rms 5×10⁻⁷),
bounded energy/momentum drift on a pinned trajectory, and the ε = 0
orbit findings below — all measured first, then pinned, because GPU float
math is not bit-reproducible across vendors. Without a WebGPU adapter the
GPU suite skips loudly; it never fakes green.

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

## GPU benchmark (Phase 3)

`npm run bench:gpu` (or open `/bench.html`) times the Float32 GPU step
against CPU Barnes–Hut at θ = 0.5 **in the same page on the same
machine** — Node can't time WebGPU, so this is a separate, browser-run
benchmark with the same honesty rules (probe > 2 s → skip, never
extrapolate). Measured on an **Apple M4 Pro (`apple metal-3`), Chromium
149 headless**:

| Bodies | CPU BH (θ=0.5) ms/step | GPU brute f32 ms/step | GPU speedup |
|-------:|--------------------------:|----------------------:|------------:|
| 2,000 | 2.36 | 0.19 | 12.5× |
| 5,000 | 7.89 | 0.46 | 17.3× |
| 10,000 | 18.47 | 0.96 | 19.2× |
| 20,000 | 42.23 | 3.58 | 11.8× |
| 50,000 | 124.03 | 20.12 | 6.2× |
| 100,000 | 275.70 | 79.06 | 3.5× |
| 200,000 | 662.60 | 310.20 | 2.1× |
| 500,000 | — (> 2 s, skipped) | 1848.90 | — |

- **Crossover: the GPU wins everywhere measured** — from 2,000 bodies up,
  the exact O(N²) sum on the GPU beats the O(N log N) approximation on
  the CPU. The margin peaks (~19×) around 10k where the GPU is saturated
  but the sum is still small, then narrows as N² catches up with
  N log N — down to 2.1× at 200k. Extrapolating the measured trends, the
  curves would cross a little past 500k; a GPU tree would be the answer
  there, and is deliberately out of scope.
- **Max bodies at 60 fps (physics only): 20,000 exact-N²** (3.58 ms/step;
  50k takes 20.1 ms, just over the 16.7 ms budget). Above the GPU's
  saturation point (~10k) the timing is textbook N²: 4× per doubling.
  Below it, a fixed ~0.2 ms submit-overhead floor dominates.
- **The 60 fps claim is physics-only, and that is a real limitation:**
  Canvas 2D cannot draw 10⁵ glow sprites per frame — at these body counts
  the renderer, not the kernel, is the frame-rate ceiling. The renderer
  is deliberately untouched in this phase; WebGL/WebGPU point rendering
  is the natural next step if the demo is ever to *show* 100k bodies at
  the rate the physics can now *compute* them.
- The energy overlay survives GPU-scale N by time-slicing: the O(N²)
  Float64 energy sample is spread across frames (~3 ms slices) from a
  consistent read-back snapshot, so it stays exact at any N — samples
  just get sparser as N² grows (a 100k sample takes a few sliced
  seconds; the overlay shows "sampling…" until E₀ exists).

## What Float64 → Float32 actually cost (measured)

The point of this phase was to measure the precision price of the GPU
port with the Float64 diagnostics built in Phase 1. The measured answer,
on the pinned disk trajectory (N = 1,025, dt = 0.002, seed 42):

- **Energy drift: nothing measurable.** CPU Float64 drifts 1.35×10⁻⁶
  after 1,000 steps; GPU Float32 drifts 1.48×10⁻⁶ (1.1×). After 10,000
  steps: 1.34×10⁻⁶ vs 1.25×10⁻⁶ — *indistinguishable*. Leapfrog's
  bounded symplectic oscillation is set by dt² truncation, which sits
  well above Float32 rounding noise at these horizons. Float32 does not
  visibly degrade energy conservation here — a genuine (and genuinely
  surprising) negative result.
- **Momentum is where Float32 shows.** The CPU brute path conserves
  momentum to 10⁻¹² of the momentum scale because pair terms cancel
  exactly; the GPU sums each body independently, so Newton's third law
  survives only to rounding: measured drift 4.6×10⁻⁸ of scale after
  1,000 steps. Bounded and physically irrelevant, but five orders of
  magnitude away from "machine precision" — the honest cost.
- **The ε = 0 exact orbits survive.** The marginally stable figure-eight
  choreography was the expected first casualty; measured, it deviates
  from the Float64 trajectory by only 1.3×10⁻⁵ after one period and
  2.4×10⁻³ after **40 periods** (~0.1% of the system size, growth
  roughly linear — no chaotic blowup on that horizon). Sun+planets
  radii stay within 1.0×10⁻⁴ of Float64 over three outer orbits.
- One-off cost on entering GPU mode: the state is quantized to Float32
  (relative 6×10⁻⁸); E₀ is re-baselined from the quantized state so the
  overlay's drift measures integration, not the upload rounding.

Numbers are from `npm run test:gpu` on the machine above; the suite pins
each one with headroom for cross-vendor Float32 reduction-order
differences, so a regression in the shader physics fails tests rather
than silently rewriting this section.

## Roadmap

- **Phase 1 (done)** — brute-force CPU baseline: correctness, tests,
  diagnostics, benchmark.
- **Phase 2 (done)** — Barnes–Hut quadtree: O(N log N) forces verified
  against the brute-force oracle; the benchmark above quantifies the
  speedup (crossover ~1–2k bodies, 8.6× at 20k), the energy-drift
  diagnostic quantifies the accuracy cost of θ. Deliberately not built:
  incremental tree updates, higher-order multipoles, adaptive timesteps.
- **Phase 3 (done)** — WebGPU compute port: the whole leapfrog step
  (kick-drift / force / kick) as Float32 WGSL compute passes behind the
  existing `Integrator` seam, CPU state demoted to a per-frame read-back
  mirror. GPU brute force beats CPU Barnes–Hut from 2k bodies up and
  holds 60 fps physics at 20k exact-N²; the Float64 diagnostics measured
  the precision cost (see above — energy drift indistinguishable,
  momentum at 10⁻⁸-scale rounding, ε = 0 orbits intact). Deliberately
  not built: a GPU tree (the >500k answer), workers/SIMD on the CPU
  path, and the renderer rewrite (WebGL/WebGPU point sprites) that the
  measured rendering bottleneck now motivates as the natural Phase 4.
