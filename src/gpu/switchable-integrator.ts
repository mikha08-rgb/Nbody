import type { Integrator } from '../physics/integrator';
import type { SimState } from '../physics/state';
import type { GpuLeapfrog } from './gpu-leapfrog';

/**
 * Delegates the Integrator seam to either the CPU leapfrog or the GPU
 * leapfrog, so Simulation itself needs no changes to support live method
 * switching (brute/Barnes–Hut switching stays inside the CPU force
 * closure, exactly as in Phase 2; this class only bridges CPU ↔ GPU).
 *
 * The DESIRED path (`wantGpu`) is tracked separately from the currently
 * ACTIVE delegate, and init() re-derives active from desired — so a
 * scenario reset lands directly on the right integrator. A fresh Float64
 * state must never round-trip through a stale GPU delegate: that would
 * quantize it to Float32 for no reason (an ε = 0 scenario's exact initial
 * conditions included).
 *
 * The two live-switch directions are asymmetric:
 *
 * - CPU → GPU is synchronous: upload the current state (quantizing the
 *   mirror to Float32 in place — the documented one-off cost of entering
 *   GPU mode) and prime accelerations.
 * - GPU → CPU waits for an async readback of the authoritative GPU state.
 *   Until it lands (typically within a frame), step() drops ticks. When
 *   the CPU takes over, the retired GPU system is DESTROYED: its buffers
 *   are freed and any still-in-flight snapshot becomes a no-op instead of
 *   overwriting a state the CPU now owns.
 *
 * Every transition and init bumps `generation`, so a superseded handoff
 * can never install its delegate late. Re-selecting GPU while a GPU→CPU
 * handoff is pending just cancels the handoff — the GPU buffers are still
 * authoritative, so there is nothing to re-upload and no precision is
 * lost to a spurious re-quantization.
 */
export class SwitchableIntegrator implements Integrator {
  readonly name = 'switchable';
  private active: Integrator;
  private wantGpu: boolean;
  /** Bumped by anything that supersedes an in-flight handoff. */
  private generation = 0;
  /** True while a GPU→CPU readback is in flight (ticks are dropped). */
  private waiting = false;

  constructor(
    private readonly cpu: Integrator,
    private gpu: GpuLeapfrog | null,
    useGpu: boolean,
  ) {
    this.wantGpu = useGpu && gpu !== null;
    this.active = this.wantGpu && gpu !== null ? gpu : cpu;
  }

  get usingGpu(): boolean {
    return this.gpu !== null && this.active === this.gpu;
  }

  /** A GPU delegate exists and its device is believed alive. */
  get gpuAvailable(): boolean {
    return this.gpu !== null;
  }

  /** Frame-loop backpressure signal; always false on the CPU path. */
  get backpressured(): boolean {
    return this.usingGpu && this.gpu !== null && this.gpu.backpressured;
  }

  /**
   * Provide the GPU delegate once async device acquisition finishes — the
   * app boots CPU-first and never blocks on WebGPU detection.
   */
  attachGpu(gpu: GpuLeapfrog): void {
    if (this.gpu === null) this.gpu = gpu;
  }

  /**
   * Record which path the NEXT init() should activate, without touching
   * the running system. resetScenario sets this before Simulation.reset,
   * which is what keeps fresh states off the stale delegate.
   */
  setDesiredMode(useGpu: boolean): void {
    this.wantGpu = useGpu && this.gpu !== null;
  }

  init(state: SimState): void {
    this.generation++; // a reset supersedes any in-flight handoff
    this.waiting = false;
    this.active = this.wantGpu && this.gpu !== null ? this.gpu : this.cpu;
    this.active.init(state);
  }

  step(state: SimState, dt: number): void {
    if (this.waiting) return; // dropped tick during GPU→CPU handoff
    this.active.step(state, dt);
  }

  /** Per-frame mirror refresh; no-op on the CPU path and mid-handoff. */
  pump(state: SimState): void {
    if (this.usingGpu && !this.waiting) this.gpu?.pumpSnapshot(state);
  }

  /** Switch paths, preserving the running system (no regeneration). */
  setUseGpu(useGpu: boolean, state: SimState): void {
    const gpu = this.gpu;
    this.wantGpu = useGpu && gpu !== null;
    if (gpu === null) return;
    this.generation++; // supersede any pending handoff, in both directions
    const generation = this.generation;

    if (useGpu) {
      if (this.active === gpu) {
        // Re-selected GPU mid-handoff: the generation bump above cancelled
        // the pending CPU installation; the GPU never stopped owning the
        // state, so just resume stepping.
        this.waiting = false;
        return;
      }
      this.waiting = false;
      gpu.init(state);
      this.active = gpu;
      return;
    }

    if (this.active === this.cpu) return;
    // CPU takes over only after a fresh full readback lands; ticks in
    // between are dropped (≤ a frame).
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
        // Retire the GPU system: free its buffers and neutralize any
        // snapshot still in flight (it must not write into a state the
        // CPU now owns).
        gpu.dispose();
      });
  }

  /**
   * Device-loss path: adopt the mirror as-is, continue on the CPU, and
   * retire the GPU delegate for good — a lost device never comes back.
   * The mirror may be a frame or two behind the lost GPU state; losing
   * those frames to a dying driver is the honest option available.
   */
  fallbackToCpu(state: SimState): void {
    this.generation++;
    this.waiting = false;
    this.wantGpu = false;
    this.gpu?.dispose();
    this.gpu = null;
    this.cpu.init(state);
    this.active = this.cpu;
  }
}
