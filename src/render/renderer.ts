import type { SimState } from '../physics/state';
import type { Camera } from './camera';
import { createGlowSprite } from './sprite';

const BACKGROUND = '#05070c';
/** On-screen sprite size (CSS px) for a body of reference mass. */
const BASE_SPRITE_PX = 9;

/**
 * Canvas 2D particle renderer: dark background, additive pre-rendered
 * glow sprites (see sprite.ts).
 */
export class Renderer {
  private readonly ctx: CanvasRenderingContext2D;
  private readonly sprite: HTMLCanvasElement;
  private cssWidth = 0;
  private cssHeight = 0;

  constructor(private readonly canvas: HTMLCanvasElement) {
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('2D canvas context unavailable');
    this.ctx = ctx;
    this.sprite = createGlowSprite();
  }

  /**
   * Match the drawing buffer to the element's CSS size × devicePixelRatio,
   * so output is crisp on high-DPI displays; all drawing then happens in
   * CSS-pixel coordinates via the transform.
   */
  resize(): { width: number; height: number } {
    const dpr = window.devicePixelRatio || 1;
    this.cssWidth = this.canvas.clientWidth;
    this.cssHeight = this.canvas.clientHeight;
    this.canvas.width = Math.round(this.cssWidth * dpr);
    this.canvas.height = Math.round(this.cssHeight * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { width: this.cssWidth, height: this.cssHeight };
  }

  draw(state: SimState, camera: Camera): void {
    const { ctx, sprite } = this;
    const { n, px, py, mass } = state;

    ctx.globalCompositeOperation = 'source-over';
    ctx.fillStyle = BACKGROUND;
    ctx.fillRect(0, 0, this.cssWidth, this.cssHeight);

    // Additive blending: overlapping glows brighten, reading as density.
    ctx.globalCompositeOperation = 'lighter';

    // Scale sprite size gently with mass so a central body reads as a star,
    // not just another particle. Reference mass = mean mass.
    let totalMass = 0;
    for (let i = 0; i < n; i++) totalMass += mass[i];
    const meanMass = totalMass / n;

    for (let i = 0; i < n; i++) {
      const sx = camera.toScreenX(px[i]);
      const sy = camera.toScreenY(py[i]);
      const size = BASE_SPRITE_PX * Math.min(4, Math.max(0.7, Math.pow(mass[i] / meanMass, 0.25)));
      const half = size / 2;
      if (sx < -half || sy < -half || sx > this.cssWidth + half || sy > this.cssHeight + half) {
        continue; // off-screen
      }
      ctx.drawImage(sprite, sx - half, sy - half, size, size);
    }

    ctx.globalCompositeOperation = 'source-over';
  }
}
