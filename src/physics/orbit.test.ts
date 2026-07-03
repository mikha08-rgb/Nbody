import { describe, expect, it } from 'vitest';
import { Leapfrog } from './leapfrog';
import { createState } from './state';

/**
 * Two equal masses on a mutual circular orbit must stay on it.
 *
 * Setup: m = 1 each, separation d = 1, ε = 0. Force balance gives
 * ω = √(2·G·m / d³) = √2, each body circling the origin at radius d/2
 * with speed ω·d/2.
 *
 * Tolerance is a spec contract: radius within 1% over 100 orbits at
 * 1,000 steps per orbit. Do not loosen it to make the test pass.
 */
describe('two-body circular orbit', () => {
  it('keeps orbital radius within 1% over 100 orbits', () => {
    const s = createState(2, 0);
    s.mass.fill(1);
    const radius = 0.5;
    const omega = Math.SQRT2;
    const speed = omega * radius;
    s.px[0] = radius;
    s.px[1] = -radius;
    s.vy[0] = speed;
    s.vy[1] = -speed;

    const stepsPerOrbit = 1000;
    const dt = (2 * Math.PI) / omega / stepsPerOrbit;
    const integrator = new Leapfrog();
    integrator.init(s);

    let maxRelErr = 0;
    for (let step = 0; step < 100 * stepsPerOrbit; step++) {
      integrator.step(s, dt);
      for (let i = 0; i < 2; i++) {
        const r = Math.hypot(s.px[i], s.py[i]);
        maxRelErr = Math.max(maxRelErr, Math.abs(r - radius) / radius);
      }
    }
    expect(maxRelErr).toBeLessThan(0.01);
  });
});
