import type { SimState } from '../physics/state';

/**
 * World ↔ screen mapping with pan and zoom.
 *
 * Screen y grows downward; world y grows upward (physics convention), so
 * the y axis flips here and nowhere else.
 */
export class Camera {
  /** World coordinates at the center of the viewport. */
  cx = 0;
  cy = 0;
  /** Zoom: CSS pixels per world unit. */
  pxPerUnit = 120;

  private viewW = 1;
  private viewH = 1;

  setViewport(width: number, height: number): void {
    this.viewW = width;
    this.viewH = height;
  }

  toScreenX(wx: number): number {
    return (wx - this.cx) * this.pxPerUnit + this.viewW / 2;
  }

  toScreenY(wy: number): number {
    return this.viewH / 2 - (wy - this.cy) * this.pxPerUnit;
  }

  /** Pan by a screen-space pixel delta (drag). */
  panByPixels(dxPix: number, dyPix: number): void {
    this.cx -= dxPix / this.pxPerUnit;
    this.cy += dyPix / this.pxPerUnit;
  }

  /** Zoom by `factor`, keeping the world point under (sx, sy) fixed. */
  zoomAt(sx: number, sy: number, factor: number): void {
    const wx = this.cx + (sx - this.viewW / 2) / this.pxPerUnit;
    const wy = this.cy + (this.viewH / 2 - sy) / this.pxPerUnit;
    this.pxPerUnit = Math.min(20000, Math.max(2, this.pxPerUnit * factor));
    this.cx = wx - (sx - this.viewW / 2) / this.pxPerUnit;
    this.cy = wy - (this.viewH / 2 - sy) / this.pxPerUnit;
  }

  /** Center on the system and zoom so the whole scenario fits comfortably. */
  fit(state: SimState): void {
    let maxR = 0;
    for (let i = 0; i < state.n; i++) {
      maxR = Math.max(maxR, Math.hypot(state.px[i], state.py[i]));
    }
    this.cx = 0;
    this.cy = 0;
    this.pxPerUnit = (0.42 * Math.min(this.viewW, this.viewH)) / Math.max(maxR, 1e-9);
  }
}

/** Wire pointer-drag panning and wheel zooming to a camera. */
export function attachCameraControls(canvas: HTMLCanvasElement, camera: Camera): void {
  let dragging = false;
  let lastX = 0;
  let lastY = 0;

  canvas.addEventListener('pointerdown', (e) => {
    dragging = true;
    lastX = e.clientX;
    lastY = e.clientY;
    canvas.setPointerCapture(e.pointerId);
    canvas.classList.add('dragging');
  });
  canvas.addEventListener('pointermove', (e) => {
    if (!dragging) return;
    camera.panByPixels(e.clientX - lastX, e.clientY - lastY);
    lastX = e.clientX;
    lastY = e.clientY;
  });
  canvas.addEventListener('pointerup', (e) => {
    dragging = false;
    canvas.releasePointerCapture(e.pointerId);
    canvas.classList.remove('dragging');
  });
  // Touch interruptions (system gestures) end a drag via pointercancel,
  // not pointerup; capture is released implicitly.
  canvas.addEventListener('pointercancel', () => {
    dragging = false;
    canvas.classList.remove('dragging');
  });
  canvas.addEventListener(
    'wheel',
    (e) => {
      e.preventDefault();
      camera.zoomAt(e.clientX, e.clientY, Math.exp(-e.deltaY * 0.0015));
    },
    { passive: false },
  );
}
