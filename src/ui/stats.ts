export interface StatsData {
  fps: number;
  /** Relative energy drift |E − E₀|/|E₀|, or null before the first sample. */
  drift: number | null;
  /** True while the time-sliced energy sampler is mid-sample. */
  sampling: boolean;
  bodies: number;
  paused: boolean;
}

/** Small read-only overlay: FPS, body count, energy drift. */
export class StatsOverlay {
  constructor(private readonly root: HTMLElement) {}

  update(d: StatsData): void {
    const fps = d.paused ? 'paused' : `${d.fps.toFixed(0)} fps`;
    const drift =
      d.drift === null
        ? d.sampling
          ? 'sampling…'
          : '—'
        : `${(d.drift * 100).toPrecision(2)}%`;
    this.root.innerHTML =
      `<div>${fps}</div>` +
      `<div>${d.bodies.toLocaleString()} bodies</div>` +
      `<div>ΔE/E₀: ${drift}</div>`;
  }
}
