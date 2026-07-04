import { configDefaults, defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    include: ['src/**/*.test.ts'],
    // *.gpu.test.ts need WebGPU and run in browser mode under
    // vitest.gpu.config.ts (`npm run test:gpu`); Node has no WebGPU.
    exclude: [...configDefaults.exclude, '**/*.gpu.test.ts'],
  },
});
