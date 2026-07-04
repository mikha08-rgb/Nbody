import type { Integrator } from '../physics/integrator';
import type { SimState } from '../physics/state';
import { MAX_SNAPSHOTS_IN_FLIGHT, NBodyGpu, NBodyPrograms } from './nbody-gpu';

/**
 * GPU-resident leapfrog — the Phase 3 Integrator.
 *
 * The seam survives the sync/async split because only READBACK is async in
 * WebGPU: encoding and submitting compute passes are ordinary synchronous
 * host calls. So step() stays a synchronous Integrator.step — it submits
 * the three KDK passes and returns; the GPU works through the queue on its
 * own time. Simulation's accumulator, the substep cap, and the frame loop
 * are untouched.
 *
 * Contract reinterpretation, documented here on purpose: while this
 * integrator is active, the AUTHORITATIVE positions/velocities/
 * accelerations live in Float32 GPU buffers. The Integrator invariant
 * ("after step(), accelerations are consistent with positions") holds on
 * those buffers. SimState's Float64 arrays become a read-back mirror,
 * refreshed once per rendered frame by pumpSnapshot() — typically one to
 * two frames behind the physics — which is what the renderer and the
 * energy diagnostics read. init() quantizes the freshly generated state to
 * Float32 in place (see NBodyGpu), so E₀ and every later sample measure
 * exactly the system the GPU integrates.
 *
 * Backpressure: if the GPU cannot keep up, completed snapshots stop coming
 * back while submits would keep piling up unboundedly. `backpressured`
 * turns true when the snapshot ring is saturated; the frame loop then
 * skips sim.advance() entirely, so simulation time slows to what the GPU
 * sustains — the same drop-excess philosophy as the substep cap, instead
 * of a death spiral of queued dispatches.
 */
export class GpuLeapfrog implements Integrator {
  readonly name = 'gpu-leapfrog';
  private readonly programs: NBodyPrograms;
  private system: NBodyGpu | null = null;
  private stepsSinceSnapshot = 0;

  constructor(device: GPUDevice) {
    this.programs = new NBodyPrograms(device);
  }

  /** Upload (quantizing the mirror in place) and prime accelerations. */
  init(state: SimState): void {
    this.system?.destroy();
    this.system = new NBodyGpu(this.programs, state);
    this.system.primeForces();
    this.stepsSinceSnapshot = 0;
  }

  step(state: SimState, dt: number): void {
    const system = this.requireSystem(state);
    system.step(dt);
    state.time += dt; // sim time stays CPU-side Float64 bookkeeping
    this.stepsSinceSnapshot++;
  }

  /** True when the snapshot ring is saturated — stop scheduling steps. */
  get backpressured(): boolean {
    return this.system !== null && this.system.pendingSnapshots >= MAX_SNAPSHOTS_IN_FLIGHT;
  }

  /**
   * Per-frame mirror refresh: request a positions+velocities snapshot if
   * any steps ran since the last request (while paused the mirror is
   * already current, so nothing is submitted).
   */
  pumpSnapshot(state: SimState): void {
    if (this.system === null || this.stepsSinceSnapshot === 0) return;
    if (this.system.requestSnapshot(state)) {
      this.stepsSinceSnapshot = 0;
    }
  }

  /**
   * Awaited full readback into `state` — the GPU→CPU handoff path (and the
   * tests'). The mirror afterwards is fresh, not one frame stale.
   */
  async readState(state: SimState, includeAccelerations = false): Promise<void> {
    await this.requireSystem(state).readState(state, includeAccelerations);
  }

  private requireSystem(state: SimState): NBodyGpu {
    if (this.system === null || this.system.n !== state.n) {
      throw new Error('GpuLeapfrog used without init() for this state');
    }
    return this.system;
  }
}
