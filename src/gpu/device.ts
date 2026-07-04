/**
 * WebGPU feature detection and device acquisition.
 *
 * Everything GPU-related hangs off this single async entry point: no
 * adapter (old browser, insecure context, headless shell, driver
 * blocklist) → null, and the app runs exactly as it did in Phases 1–2
 * with the GPU option disabled in the UI.
 */
export interface GpuContext {
  readonly device: GPUDevice;
  /**
   * Human-readable adapter description (e.g. "apple metal-3") for the
   * benchmark page and README provenance. Fields are best-effort — some
   * browsers redact parts of GPUAdapterInfo.
   */
  readonly adapterLabel: string;
  /**
   * Resolves if the device is ever lost (driver reset, GPU removed). The
   * app treats this as "WebGPU just became unavailable": fall back to the
   * CPU path and disable the option.
   */
  readonly lost: Promise<GPUDeviceLostInfo>;
}

export async function acquireGpu(): Promise<GpuContext | null> {
  if (!('gpu' in navigator)) return null;
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return null;
    const device = await adapter.requestDevice();
    const info = adapter.info;
    const adapterLabel =
      [info.vendor, info.architecture, info.device, info.description]
        .filter((part) => part.length > 0)
        .join(' ') || 'unknown adapter';
    return { device, adapterLabel, lost: device.lost };
  } catch {
    // A throwing requestAdapter/requestDevice is the same situation as a
    // missing adapter: WebGPU is not usable here.
    return null;
  }
}
