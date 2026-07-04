import { describe, expect, it } from 'vitest';
import { computeAccelerations } from '../physics/forces';
import { createState } from '../physics/state';
import { diskGalaxy } from '../scenarios/disk-galaxy';
import { GpuLeapfrog } from './gpu-leapfrog';
import { acquireGpuForTests } from './gpu-test-utils';

const ctx = await acquireGpuForTests();

/**
 * GPU force kernel vs the CPU brute-force oracle on the seeded disk.
 *
 * PINNED SETUP — do not loosen tolerances or reshuffle the seed (see the
 * test policy in CLAUDE.md). GPU float math is not bit-reproducible across
 * vendors, so unlike the CPU θ = 0 contract (1e-12) this is a tolerance
 * comparison, justified from Float32 arithmetic:
 *
 * The reference is the exact Float64 kernel evaluated on the SAME
 * Float32-quantized inputs the GPU sees (GpuLeapfrog.init quantizes the
 * state in place), so the comparison isolates the shader's f32 arithmetic
 * — input representation error is measured separately by the drift tests.
 * Summing ~N f32 terms with unit machine epsilon 2^-24 ≈ 6e-8 and mild
 * cancellation gives per-body errors of order √N·ε ≈ 3e-6 at N = 2049; a
 * kernel bug (wrong ε, missed tile, self-interaction) shows up orders of
 * magnitude above that.
 *
 * Error metric: |Δa| relative to max(|a_ref|, RMS|a|) — near-cancelling
 * accelerations (a body the whole disk pulls on almost symmetrically)
 * would make a pure relative error blow up on physically tiny quantities.
 */
const N = 2048; // disk particles; the scenario adds the central body
const SEED = 42;

// Measured on Apple M-series (metal-3), Chromium 149: rms 5.4e-7, max
// 3.4e-6 — right at the √N·ε estimate. Pinned with ~5× headroom for
// vendor-to-vendor reduction-order differences.
const RMS_TOL = 2.5e-6;
const MAX_TOL = 2e-5;

describe.skipIf(ctx === null)('GPU force kernel parity (pinned disk, N=2048)', () => {
  it('matches Float64 brute force on quantized inputs within f32 error', async () => {
    const state = diskGalaxy.generate(N, SEED);
    const gpu = new GpuLeapfrog(ctx!.device);
    gpu.init(state); // uploads + quantizes the mirror in place
    await gpu.readState(state, true); // pulls back the primed accelerations

    // Float64 reference on the quantized inputs. ε enters the GPU kernel
    // as f32(ε²); √ of that in Float64 reproduces it to 1 ulp — far below
    // anything this comparison can resolve.
    const ref = createState(state.n, Math.sqrt(Math.fround(state.eps * state.eps)));
    ref.px.set(state.px);
    ref.py.set(state.py);
    ref.mass.set(state.mass);
    computeAccelerations(ref);

    let norm2 = 0;
    for (let i = 0; i < ref.n; i++) {
      norm2 += ref.ax[i] * ref.ax[i] + ref.ay[i] * ref.ay[i];
    }
    const rmsAcc = Math.sqrt(norm2 / ref.n);

    let sumErr2 = 0;
    let maxErr = 0;
    for (let i = 0; i < ref.n; i++) {
      const dx = state.ax[i] - ref.ax[i];
      const dy = state.ay[i] - ref.ay[i];
      const scale = Math.max(Math.hypot(ref.ax[i], ref.ay[i]), rmsAcc);
      const err = Math.hypot(dx, dy) / scale;
      sumErr2 += err * err;
      maxErr = Math.max(maxErr, err);
    }
    const rmsErr = Math.sqrt(sumErr2 / ref.n);

    console.log(
      `force parity (N=${ref.n}): rms=${rmsErr.toExponential(2)} max=${maxErr.toExponential(2)}`,
    );
    expect(rmsErr).toBeLessThan(RMS_TOL);
    expect(maxErr).toBeLessThan(MAX_TOL);
  });
});
