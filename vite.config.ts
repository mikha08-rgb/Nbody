import { configDefaults, defineConfig } from 'vitest/config';

export default defineConfig({
  build: {
    rollupOptions: {
      // Two pages: the simulator and the GPU benchmark (/bench.html).
      // Plain root-relative paths — URL.pathname percent-encodes (a
      // checkout under "My Projects" broke the build) and keeps the
      // /C:/ prefix on Windows; Vite resolves these against the root.
      input: {
        main: 'index.html',
        bench: 'bench.html',
      },
    },
  },
  test: {
    include: ['src/**/*.test.ts'],
    // *.gpu.test.ts need WebGPU and run in browser mode under
    // vitest.gpu.config.ts (`npm run test:gpu`); Node has no WebGPU.
    exclude: [...configDefaults.exclude, '**/*.gpu.test.ts'],
  },
});
