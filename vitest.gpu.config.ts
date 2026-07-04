import { playwright } from '@vitest/browser-playwright';
import { defineConfig } from 'vitest/config';

/**
 * GPU test suite: `npm run test:gpu`.
 *
 * Vitest normally runs in Node, which has no WebGPU — these tests run in
 * real Chromium via browser mode instead. Two environment facts matter:
 *
 * - Playwright's default headless browser is the "headless shell", which
 *   has NO WebGPU. `channel: 'chromium'` selects the full Chromium build
 *   in new-headless mode, which exposes the real GPU (Metal on macOS).
 * - If no adapter exists anyway (CI without a GPU), every suite skips
 *   with a loud console warning rather than failing — see
 *   src/gpu/gpu-test-utils.ts.
 */
export default defineConfig({
  test: {
    include: ['src/**/*.gpu.test.ts'],
    // GPU pipelines compile on first use and some runs integrate tens of
    // thousands of steps; generous timeouts keep slow adapters honest.
    testTimeout: 120_000,
    hookTimeout: 120_000,
    browser: {
      enabled: true,
      headless: true,
      screenshotFailures: false,
      provider: playwright({ launchOptions: { channel: 'chromium' } }),
      instances: [{ browser: 'chromium' }],
    },
  },
});
