import type { Integrator } from './integrator';
import type { SimState } from './state';

/** Physics ticks per real second (at normal speed, one tick integrates `dt`). */
const TICK_RATE = 60;

/**
 * Hard cap on physics ticks per frame. If a frame arrives owing more ticks
 * than this, the excess accumulated time is DROPPED: the sim slows down
 * rather than scheduling ever more work per frame — the death spiral that
 * would freeze a lagging tab.
 */
const MAX_SUBSTEPS = 6;

/**
 * Fixed-timestep driver, decoupled from the render loop.
 *
 * Physics advances in fixed ticks fed by an accumulator of real elapsed
 * time; the renderer just draws whatever state exists each frame. dt is
 * never derived from requestAnimationFrame deltas, so physics results are
 * independent of display refresh rate and frame hiccups.
 */
export class Simulation {
  running = true;

  /**
   * Simulation time integrated per tick. Mutable live from the UI slider;
   * note that changing dt mid-run breaks leapfrog's symplectic energy
   * conservation (see leapfrog.ts) — a small one-off energy shift is
   * expected, not a bug.
   */
  dt: number;

  private accumulator = 0;

  constructor(
    public state: SimState,
    public readonly integrator: Integrator,
    dt: number,
  ) {
    this.dt = dt;
    this.integrator.init(state);
  }

  /** Swap in a freshly generated state (scenario change / reset). */
  reset(state: SimState, dt: number): void {
    this.state = state;
    this.dt = dt;
    this.accumulator = 0;
    this.integrator.init(state);
  }

  /**
   * Feed real elapsed seconds; runs 0..MAX_SUBSTEPS physics ticks.
   * Returns the number of ticks executed (0 while paused).
   */
  advance(elapsedSeconds: number): number {
    if (!this.running) return 0;
    // Clamp monster gaps (backgrounded tab) before they reach the cap.
    this.accumulator += Math.min(elapsedSeconds, 0.25);
    const tick = 1 / TICK_RATE;
    let steps = 0;
    while (this.accumulator >= tick && steps < MAX_SUBSTEPS) {
      this.integrator.step(this.state, this.dt);
      this.accumulator -= tick;
      steps++;
    }
    if (this.accumulator >= tick) {
      this.accumulator = 0; // cap hit → drop the excess, don't spiral
    }
    return steps;
  }
}
