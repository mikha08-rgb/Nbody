import { configDefaults, defineConfig } from 'vitest/config';

export default defineConfig({
  build: {
    rollupOptions: {
      // Two pages: the simulator and the GPU benchmark (/bench.html).
      input: {
        main: new URL('./index.html', import.meta.url).pathname,
        bench: new URL('./bench.html', import.meta.url).pathname,
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
