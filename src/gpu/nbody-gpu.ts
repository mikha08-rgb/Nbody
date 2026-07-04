import type { SimState } from '../physics/state';
import kernels from './nbody-kernels.wgsl?raw';

/**
 * GPU-side simulation core: Float32 storage buffers mirroring the SimState
 * structure-of-arrays layout, plus the three leapfrog compute passes from
 * nbody-kernels.wgsl.
 *
 * Split in two because their lifetimes differ:
 *
 * - NBodyPrograms — shader module, bind-group layout, pipelines. Depends
 *   only on the device; created once at startup and reused across every
 *   scenario reset (pipeline compilation is the expensive part).
 * - NBodyGpu — the buffers for one system of fixed n. Rebuilt per
 *   scenario (re)generation, destroyed on the next.
 *
 * ## Who owns the truth
 *
 * While a system is GPU-resident, the storage buffers are authoritative;
 * the CPU SimState becomes a read-back mirror refreshed by requestSnapshot
 * (per rendered frame, ring-buffered, non-blocking) or readState (one-off,
 * awaited — tests and CPU handoff). Constructing an NBodyGpu quantizes the
 * SimState arrays to Float32 IN PLACE, so the mirror, the diagnostics, and
 * any later CPU handoff all see exactly the system the GPU integrates —
 * entering GPU mode deliberately costs one Float64→Float32 rounding, part
 * of the precision story, not hidden.
 */

/**
 * Threads per workgroup. THE single source: injected into the WGSL as a
 * pipeline-creation override constant, so the shader's indexing/tiling
 * and the dispatch count below can never disagree (a silent mismatch
 * would leave part of the system frozen with no error raised).
 */
export const WORKGROUP_SIZE = 256;

/**
 * Snapshot ring size. Two in flight means the mirror lags the physics by
 * at most ~two rendered frames; when both slots are busy the GPU is
 * falling behind and requestSnapshot returns false — the caller's
 * backpressure signal to stop scheduling steps (the sim slows down instead
 * of queueing unbounded GPU work, same philosophy as the substep cap).
 */
export const MAX_SNAPSHOTS_IN_FLIGHT = 2;

export class NBodyPrograms {
  readonly bindGroupLayout: GPUBindGroupLayout;
  readonly kickDrift: GPUComputePipeline;
  readonly force: GPUComputePipeline;
  readonly kick: GPUComputePipeline;

  constructor(readonly device: GPUDevice) {
    const module = device.createShaderModule({ label: 'nbody-kernels', code: kernels });
    const buffer = (
      binding: number,
      type: GPUBufferBindingType,
    ): GPUBindGroupLayoutEntry => ({
      binding,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type },
    });
    // Bindings 0–7 as declared in nbody-kernels.wgsl. One shared layout for
    // all three pipelines, so one bind group serves the whole step.
    this.bindGroupLayout = device.createBindGroupLayout({
      label: 'nbody-buffers',
      entries: [
        buffer(0, 'uniform'),
        buffer(1, 'storage'),
        buffer(2, 'storage'),
        buffer(3, 'storage'),
        buffer(4, 'storage'),
        buffer(5, 'storage'),
        buffer(6, 'storage'),
        buffer(7, 'read-only-storage'),
      ],
    });
    const layout = device.createPipelineLayout({
      label: 'nbody-pipeline-layout',
      bindGroupLayouts: [this.bindGroupLayout],
    });
    const pipeline = (entryPoint: string): GPUComputePipeline =>
      device.createComputePipeline({
        label: `nbody-${entryPoint}`,
        layout,
        compute: { module, entryPoint, constants: { WORKGROUP_SIZE } },
      });
    this.kickDrift = pipeline('kick_drift');
    this.force = pipeline('force');
    this.kick = pipeline('kick');
  }
}

interface SnapshotSlot {
  readonly buffer: GPUBuffer;
  inFlight: boolean;
}

export class NBodyGpu {
  readonly n: number;
  private readonly device: GPUDevice;
  private readonly bindGroup: GPUBindGroup;
  private readonly posX: GPUBuffer;
  private readonly posY: GPUBuffer;
  private readonly velX: GPUBuffer;
  private readonly velY: GPUBuffer;
  private readonly accX: GPUBuffer;
  private readonly accY: GPUBuffer;
  private readonly mass: GPUBuffer;
  private readonly params: GPUBuffer;
  private readonly workgroups: number;
  private readonly snapshots: SnapshotSlot[] = [];
  private lastDt = Number.NaN;
  private pending = 0;
  private destroyed = false;

  /**
   * Uploads `state` to fresh GPU buffers, quantizing the SimState arrays
   * (positions, velocities, masses) to Float32 in place — see the module
   * comment. Accelerations start zeroed; call primeForces() to establish
   * the Integrator invariant before the first step.
   */
  constructor(
    private readonly programs: NBodyPrograms,
    state: SimState,
  ) {
    const { device } = programs;
    this.device = device;
    this.n = state.n;
    this.workgroups = Math.ceil(state.n / WORKGROUP_SIZE);

    const bytes = state.n * Float32Array.BYTES_PER_ELEMENT;
    const storage = (label: string): GPUBuffer =>
      device.createBuffer({
        label: `nbody-${label}`,
        size: bytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });
    this.posX = storage('pos-x');
    this.posY = storage('pos-y');
    this.velX = storage('vel-x');
    this.velY = storage('vel-y');
    this.accX = storage('acc-x');
    this.accY = storage('acc-y');
    this.mass = storage('mass');
    this.params = device.createBuffer({
      label: 'nbody-params',
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const scratch = new Float32Array(state.n);
    const upload = (src: Float64Array, dst: GPUBuffer): void => {
      scratch.set(src); // Float64 → Float32 rounding happens here
      src.set(scratch); // …and is written back so the mirror matches exactly
      device.queue.writeBuffer(dst, 0, scratch);
    };
    upload(state.px, this.posX);
    upload(state.py, this.posY);
    upload(state.vx, this.velX);
    upload(state.vy, this.velY);
    upload(state.mass, this.mass);

    // params.n and eps² never change for this system; dt is (re)written by
    // step() whenever the UI slider moves.
    const header = new ArrayBuffer(16);
    new Uint32Array(header, 0, 1)[0] = state.n;
    new Float32Array(header, 4, 1)[0] = state.eps * state.eps;
    device.queue.writeBuffer(this.params, 0, header);

    this.bindGroup = device.createBindGroup({
      label: 'nbody-bind-group',
      layout: programs.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.params } },
        { binding: 1, resource: { buffer: this.posX } },
        { binding: 2, resource: { buffer: this.posY } },
        { binding: 3, resource: { buffer: this.velX } },
        { binding: 4, resource: { buffer: this.velY } },
        { binding: 5, resource: { buffer: this.accX } },
        { binding: 6, resource: { buffer: this.accY } },
        { binding: 7, resource: { buffer: this.mass } },
      ],
    });
  }

  /** Snapshots submitted but not yet landed in the mirror (0–2). */
  get pendingSnapshots(): number {
    return this.pending;
  }

  /**
   * One force dispatch to populate the acceleration buffers from the
   * current positions — the GPU counterpart of Integrator.init.
   */
  primeForces(): void {
    const encoder = this.device.createCommandEncoder({ label: 'nbody-prime' });
    this.encodePass(encoder, this.programs.force);
    this.device.queue.submit([encoder.finish()]);
  }

  /**
   * One leapfrog step: kick-drift, force, kick — encoded and submitted
   * synchronously (host-side, this never blocks; the GPU works through the
   * queue asynchronously). Exactly one force dispatch per step.
   */
  step(dt: number): void {
    if (dt !== this.lastDt) {
      const f = new Float32Array([dt, 0.5 * dt]);
      this.device.queue.writeBuffer(this.params, 8, f);
      this.lastDt = dt;
    }
    const encoder = this.device.createCommandEncoder({ label: 'nbody-step' });
    this.encodePass(encoder, this.programs.kickDrift);
    this.encodePass(encoder, this.programs.force);
    this.encodePass(encoder, this.programs.kick);
    this.device.queue.submit([encoder.finish()]);
  }

  /**
   * Non-blocking mirror refresh: copy positions+velocities into a ring
   * staging buffer and, when the map resolves (typically next frame),
   * write them into `state` (Float32 values promoted to the Float64
   * arrays). Returns false — the backpressure signal — when the ring is
   * saturated (see MAX_SNAPSHOTS_IN_FLIGHT).
   *
   * Positions and velocities are copied in the same command buffer, after
   * whole submitted steps, so a landed snapshot is always a consistent
   * post-step (x, v) pair — exactly what the energy diagnostic needs.
   */
  requestSnapshot(state: SimState): boolean {
    if (this.destroyed || state.n !== this.n) return false;
    let slot = this.snapshots.find((s) => !s.inFlight);
    if (!slot) {
      if (this.snapshots.length >= MAX_SNAPSHOTS_IN_FLIGHT) return false;
      slot = {
        buffer: this.device.createBuffer({
          label: 'nbody-snapshot',
          size: this.n * 4 * Float32Array.BYTES_PER_ELEMENT,
          usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        }),
        inFlight: false,
      };
      this.snapshots.push(slot);
    }

    const bytes = this.n * Float32Array.BYTES_PER_ELEMENT;
    const encoder = this.device.createCommandEncoder({ label: 'nbody-snapshot' });
    encoder.copyBufferToBuffer(this.posX, 0, slot.buffer, 0, bytes);
    encoder.copyBufferToBuffer(this.posY, 0, slot.buffer, bytes, bytes);
    encoder.copyBufferToBuffer(this.velX, 0, slot.buffer, 2 * bytes, bytes);
    encoder.copyBufferToBuffer(this.velY, 0, slot.buffer, 3 * bytes, bytes);
    this.device.queue.submit([encoder.finish()]);

    slot.inFlight = true;
    this.pending++;
    const landed = slot;
    landed.buffer
      .mapAsync(GPUMapMode.READ)
      .then(() => {
        // Destroyed between map resolution and this microtask: drop it.
        if (this.destroyed) return;
        const view = new Float32Array(landed.buffer.getMappedRange());
        const n = this.n;
        state.px.set(view.subarray(0, n));
        state.py.set(view.subarray(n, 2 * n));
        state.vx.set(view.subarray(2 * n, 3 * n));
        state.vy.set(view.subarray(3 * n, 4 * n));
        landed.buffer.unmap();
      })
      .catch(() => undefined) // mapAsync rejects if destroy() raced the copy
      .finally(() => {
        landed.inFlight = false;
        this.pending--;
      });
    return true;
  }

  /**
   * One-off awaited readback of the full GPU state into `state` — used by
   * the tests and by the GPU→CPU method handoff, where a fresh, complete
   * mirror matters more than avoiding a round-trip stall.
   */
  async readState(state: SimState, includeAccelerations = false): Promise<void> {
    if (state.n !== this.n) {
      throw new Error(`readState: state.n=${state.n} does not match GPU system n=${this.n}`);
    }
    const bytes = this.n * Float32Array.BYTES_PER_ELEMENT;
    const sources = includeAccelerations
      ? [this.posX, this.posY, this.velX, this.velY, this.accX, this.accY]
      : [this.posX, this.posY, this.velX, this.velY];
    const staging = this.device.createBuffer({
      label: 'nbody-readback',
      size: bytes * sources.length,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const encoder = this.device.createCommandEncoder({ label: 'nbody-readback' });
    sources.forEach((src, k) => encoder.copyBufferToBuffer(src, 0, staging, k * bytes, bytes));
    this.device.queue.submit([encoder.finish()]);

    await staging.mapAsync(GPUMapMode.READ);
    const view = new Float32Array(staging.getMappedRange());
    const n = this.n;
    const targets = includeAccelerations
      ? [state.px, state.py, state.vx, state.vy, state.ax, state.ay]
      : [state.px, state.py, state.vx, state.vy];
    targets.forEach((dst, k) => dst.set(view.subarray(k * n, (k + 1) * n)));
    staging.unmap();
    staging.destroy();
  }

  /** Release all GPU buffers; in-flight snapshots resolve as no-ops. */
  destroy(): void {
    this.destroyed = true;
    const buffers = [
      this.posX,
      this.posY,
      this.velX,
      this.velY,
      this.accX,
      this.accY,
      this.mass,
      this.params,
    ];
    for (const b of buffers) b.destroy();
    for (const s of this.snapshots) s.buffer.destroy();
  }

  private encodePass(encoder: GPUCommandEncoder, pipeline: GPUComputePipeline): void {
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(this.workgroups);
    pass.end();
  }
}
