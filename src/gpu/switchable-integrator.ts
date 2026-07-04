import type { Integrator } from '../physics/integrator';
import type { SimState } from '../physics/state';
import type { GpuLeapfrog } from './gpu-leapfrog';

/**
 * Delegates the Integrator seam to either the CPU leapfrog or the GPU
 * leapfrog, so Simulation itself needs no changes to support live method
 * switching (brute/Barnes–Hut switching stays inside the CPU force
 * closure, exactly as in Phase 2; this class only bridges CPU ↔ GPU).
 *
 * The two handoff directions are asymmetric:
 *
 * - CPU → GPU is synchronous: upload the current state (quantizing the
 *   mirror to Float32 in place — a deliberate, documented one-off
 *   precision cost of entering GPU mode) and prime accelerations.
 * - GPU → CPU must wait for an async readback of the authoritative GPU
 *   state. Until it lands (typically within a frame), step() drops ticks —
 *   simulation time briefly stands still rather than integrating from a
 *   stale mirror. A generation counter cancels the handoff if a scenario
 *   reset gets there first.
 */
export class SwitchableIntegrator implements Integrator {
  readonly name = 'switchable';
  private active: Integrator;
  /** Bumped by anything that invalidates an in-flight handoff. */
  private generation = 0;
  /** True while a GPU→CPU readback is in flight (ticks are dropped). */
  private waiting = false;

  constructor(
    private readonly cpu: Integrator,
    private readonly gpu: GpuLeapfrog | null,
    useGpu: boolean,
  ) {
    this.active = useGpu && this.gpu !== null ? this.gpu : cpu;
  }

  get usingGpu(): boolean {
    return this.gpu !== null && this.active === this.gpu;
  }

  /** Frame-loop backpressure signal; always false on the CPU path. */
  get backpressured(): boolean {
    return this.usingGpu && this.gpu !== null && this.gpu.backpressured;
  }

  init(state: SimState): void {
    this.generation++; // a reset supersedes any in-flight handoff
    this.waiting = false;
    this.active.init(state);
  }

  step(state: SimState, dt: number): void {
    if (this.waiting) return; // dropped tick during GPU→CPU handoff
    this.active.step(state, dt);
  }

  /** Per-frame mirror refresh; no-op on the CPU path. */
  pump(state: SimState): void {
    if (this.usingGpu) this.gpu?.pumpSnapshot(state);
  }

  /** Switch paths, preserving the running system (no regeneration). */
  setUseGpu(useGpu: boolean, state: SimState): void {
    const gpu = this.gpu;
    if (gpu === null || useGpu === this.usingGpu) return;
    this.generation++;
    const generation = this.generation;
    if (useGpu) {
      this.waiting = false;
      gpu.init(state);
      this.active = gpu;
      return;
    }
    this.waiting = true;
    gpu
      .readState(state)
      // Readback can only fail if the device died mid-switch; the ≤2-frame
      // stale mirror is then the best surviving record of the system.
      .catch(() => undefined)
      .then(() => {
        if (generation !== this.generation) return;
        this.cpu.init(state);
        this.active = this.cpu;
        this.waiting = false;
      });
  }

  /**
   * Device-loss path: adopt the mirror as-is and continue on the CPU. The
   * mirror may be a frame or two behind the lost GPU state — losing those
   * frames to a dying driver is the honest option available.
   */
  fallbackToCpu(state: SimState): void {
    this.generation++;
    this.waiting = false;
    this.cpu.init(state);
    this.active = this.cpu;
  }
}
