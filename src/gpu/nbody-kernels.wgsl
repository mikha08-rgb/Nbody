// N-body leapfrog in Float32 compute shaders — the Phase 3 GPU port.
//
// Physics contract, identical to the CPU path (src/physics/forces.ts and
// leapfrog.ts): G = 1 natural units, Plummer softening with per-scenario
// eps² (params.eps2), kick–drift–kick with exactly ONE O(N²) force
// dispatch per step. Accelerations persist in their storage buffers
// between steps, so the closing kick of step n reuses them as the opening
// kick of step n+1 — the Phase 1 acceleration-reuse invariant, verbatim.
//
// Precision honesty: everything here is f32 on purpose — measuring what
// that costs versus the Float64 CPU baseline is the point of the phase.
// The arithmetic mirrors the CPU kernel expression for expression
// (1.0 / (r2 * sqrt(r2))), with no inverseSqrt, no fast-math
// rearrangement, no Kahan tricks to flatter the drift numbers. Per-thread
// accumulation order is fixed (tiles in order, lanes in order), so a
// given machine reproduces its own trajectories; bit-reproducibility
// ACROSS GPUs is not promised — tests compare against the CPU oracle with
// Float32-justified tolerances instead of pinning bit-exact trajectories.
//
// Buffers are one array<f32> per SimState array — the same
// structure-of-arrays layout the CPU uses, planned since Phase 1.

struct Params {
  n: u32,
  eps2: f32,
  dt: f32,
  half_dt: f32,
}

// Single-sourced from WORKGROUP_SIZE in nbody-gpu.ts, injected at
// pipeline creation via GPUProgrammableStage.constants — a silent
// mismatch would freeze half the bodies with no error (dispatch count is
// computed on the TS side). The value below is only a default that keeps
// this file valid standalone; tune the size in nbody-gpu.ts.
override WORKGROUP_SIZE: u32 = 256u;

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> pos_x: array<f32>;
@group(0) @binding(2) var<storage, read_write> pos_y: array<f32>;
@group(0) @binding(3) var<storage, read_write> vel_x: array<f32>;
@group(0) @binding(4) var<storage, read_write> vel_y: array<f32>;
@group(0) @binding(5) var<storage, read_write> acc_x: array<f32>;
@group(0) @binding(6) var<storage, read_write> acc_y: array<f32>;
@group(0) @binding(7) var<storage, read> mass: array<f32>;

// Opening half-kick and drift, fused into one pass exactly like the CPU
// leapfrog: v += a·dt/2 (accelerations carried over from the previous
// step's closing force evaluation), then x += v·dt.
@compute @workgroup_size(WORKGROUP_SIZE)
fn kick_drift(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.n) {
    return;
  }
  let vx = vel_x[i] + params.half_dt * acc_x[i];
  let vy = vel_y[i] + params.half_dt * acc_y[i];
  vel_x[i] = vx;
  vel_y[i] = vy;
  pos_x[i] += params.dt * vx;
  pos_y[i] += params.dt * vy;
}

// Closing half-kick; the accelerations stay in place for the next step's
// opening kick.
@compute @workgroup_size(WORKGROUP_SIZE)
fn kick(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.n) {
    return;
  }
  vel_x[i] += params.half_dt * acc_x[i];
  vel_y[i] += params.half_dt * acc_y[i];
}

// Brute-force O(N²) accelerations with the classic tiled reduction (GPU
// Gems 3, ch. 31): the workgroup cooperatively stages 256 source bodies
// in workgroup memory, then every thread accumulates its own target body
// against the staged tile. One thread per target body, one buffer read
// per source body per WORKGROUP, not per thread.
var<workgroup> tile_x: array<f32, WORKGROUP_SIZE>;
var<workgroup> tile_y: array<f32, WORKGROUP_SIZE>;
var<workgroup> tile_m: array<f32, WORKGROUP_SIZE>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn force(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let i = gid.x;
  // Threads past n still run the loop: workgroupBarrier() must be reached
  // by every thread of the workgroup (uniform control flow), so inactive
  // threads only skip the loads and the accumulation, never the barriers.
  let is_body = i < params.n;
  var xi = 0.0;
  var yi = 0.0;
  if (is_body) {
    xi = pos_x[i];
    yi = pos_y[i];
  }

  var ax = 0.0;
  var ay = 0.0;
  let tile_count = (params.n + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;
  for (var t = 0u; t < tile_count; t++) {
    let j = t * WORKGROUP_SIZE + lid.x;
    if (j < params.n) {
      tile_x[lid.x] = pos_x[j];
      tile_y[lid.x] = pos_y[j];
      tile_m[lid.x] = mass[j];
    }
    workgroupBarrier();

    if (is_body) {
      // The last tile is partial: bound the scan by the real body count
      // instead of padding with fake bodies (a zero-mass body at a real
      // body's position would still produce 0·∞ = NaN when eps = 0).
      let tile_len = min(WORKGROUP_SIZE, params.n - t * WORKGROUP_SIZE);
      let base = t * WORKGROUP_SIZE;
      for (var k = 0u; k < tile_len; k++) {
        // Skip self-interaction by index, like the CPU kernel's pair loop:
        // with eps = 0 the self-term is 0/0, not 0.
        if (base + k != i) {
          let dx = tile_x[k] - xi;
          let dy = tile_y[k] - yi;
          let r2 = dx * dx + dy * dy + params.eps2;
          // Same expression as forces.ts: (r² + ε²)^(−3/2) via one sqrt
          // and one division. G = 1 in natural units, so m·dx·invR3 is the
          // whole term.
          let inv_r3 = 1.0 / (r2 * sqrt(r2));
          ax += tile_m[k] * dx * inv_r3;
          ay += tile_m[k] * dy * inv_r3;
        }
      }
    }
    workgroupBarrier();
  }

  if (is_body) {
    acc_x[i] = ax;
    acc_y[i] = ay;
  }
}
