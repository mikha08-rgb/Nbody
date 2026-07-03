/**
 * Pre-rendered glow sprite.
 *
 * Canvas radial gradients are expensive; building one PER PARTICLE PER
 * FRAME would dominate the frame budget long before physics does. Instead
 * we rasterize a single radial-gradient "star" to an offscreen canvas once
 * at startup, and the renderer stamps it with drawImage (a cheap blit)
 * under `globalCompositeOperation: 'lighter'` so overlapping glows add up.
 */
export const SPRITE_SIZE = 64;

export function createGlowSprite(): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = SPRITE_SIZE;
  canvas.height = SPRITE_SIZE;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('2D canvas context unavailable');

  const c = SPRITE_SIZE / 2;
  const gradient = ctx.createRadialGradient(c, c, 0, c, c, c);
  // Hot white core falling off through blue-white haze to transparent.
  gradient.addColorStop(0, 'rgba(255, 255, 255, 1)');
  gradient.addColorStop(0.12, 'rgba(228, 238, 255, 0.9)');
  gradient.addColorStop(0.35, 'rgba(160, 190, 255, 0.28)');
  gradient.addColorStop(0.7, 'rgba(110, 150, 240, 0.07)');
  gradient.addColorStop(1, 'rgba(90, 130, 230, 0)');
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, SPRITE_SIZE, SPRITE_SIZE);
  return canvas;
}
