import { diskGalaxy } from './disk-galaxy';
import { figureEight } from './figure-eight';
import { sunPlanets } from './sun-planets';
import type { Scenario } from './types';

/**
 * Scenario registry. Adding a scenario = one new file exporting a Scenario
 * plus one entry here; order is the UI's display order.
 */
export const scenarios: readonly Scenario[] = [diskGalaxy, figureEight, sunPlanets];

export function getScenario(id: string): Scenario {
  const s = scenarios.find((sc) => sc.id === id);
  if (!s) throw new Error(`Unknown scenario: ${id}`);
  return s;
}
