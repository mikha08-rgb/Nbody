/**
 * WebGPU feature detection and device acquisition.
 *
 * Everything GPU-related hangs off this single async entry point: no
 * adapter (old browser, insecure context, headless shell, driver
 * blocklist) → null, and the app runs exactly as it did in Phases 1–2
 * with the GPU option disabled in the UI. Callers must not block app
 * boot on this promise — requestAdapter has no spec-mandated deadline.
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
  let adapter: GPUAdapter | null;
  let device: GPUDevice;
  try {
    // Prefer the discrete GPU on dual-GPU machines: this one adapter is
    // what both the app's GPU mode and the published benchmark numbers
    // run on, and defaulting to the low-power iGPU would silently skew
    // anyone reproducing the README's crossover.
    adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (adapter === null) return null;
    device = await adapter.requestDevice();
  } catch {
    // A throwing requestAdapter/requestDevice is the same situation as a
    // missing adapter: WebGPU is not usable here.
    return null;
  }
  // The synchronous GPUAdapter.info attribute is newer than WebGPU itself
  // (Chromium 128+; earlier engines had only the since-removed
  // requestAdapterInfo, or nothing). @webgpu/types declares it
  // non-optional, hence the structural cast — a missing LABEL must not
  // discard a working DEVICE.
  const info = (adapter as { info?: Partial<GPUAdapterInfo> }).info;
  const parts =
    info === undefined ? [] : [info.vendor, info.architecture, info.device, info.description];
  const adapterLabel =
    parts.filter((part): part is string => typeof part === 'string' && part.length > 0).join(' ') ||
    'unknown adapter';
  return { device, adapterLabel, lost: device.lost };
}
