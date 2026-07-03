import type { Scenario } from '../scenarios/types';

export interface ControlsConfig {
  scenarios: readonly Scenario[];
  activeId: string;
  count: number;
  dt: number;
  /** Toggle play/pause; returns the new running state. */
  onTogglePlay(): boolean;
  onReset(): void;
  onScenarioChange(id: string): void;
  /** Fires on slider release — a new count regenerates the scenario. */
  onCountChange(n: number): void;
  /** Fires live while dragging — dt applies to the running sim. */
  onDtChange(dt: number): void;
}

export interface Controls {
  /** Sync widgets after a scenario switch (count bounds, dt, enablement). */
  syncScenario(scenario: Scenario, count: number, dt: number): void;
}

function row(label: string, ...children: HTMLElement[]): HTMLDivElement {
  const div = document.createElement('div');
  div.className = 'row';
  const lab = document.createElement('label');
  lab.textContent = label;
  div.append(lab, ...children);
  return div;
}

function slider(min: number, max: number, step: number, value: number): HTMLInputElement {
  const input = document.createElement('input');
  input.type = 'range';
  input.min = String(min);
  input.max = String(max);
  input.step = String(step);
  input.value = String(value);
  return input;
}

/** Minimal DOM control panel — play/pause, reset, scenario, N, dt. */
export function buildControls(root: HTMLElement, cfg: ControlsConfig): Controls {
  const playBtn = document.createElement('button');
  playBtn.textContent = 'Pause';
  playBtn.addEventListener('click', () => {
    playBtn.textContent = cfg.onTogglePlay() ? 'Pause' : 'Play';
  });

  const resetBtn = document.createElement('button');
  resetBtn.textContent = 'Reset';
  resetBtn.addEventListener('click', () => cfg.onReset());

  const select = document.createElement('select');
  for (const sc of cfg.scenarios) {
    const option = document.createElement('option');
    option.value = sc.id;
    option.textContent = sc.label;
    select.append(option);
  }
  select.value = cfg.activeId;
  select.addEventListener('change', () => cfg.onScenarioChange(select.value));

  const countOut = document.createElement('output');
  const countSlider = slider(100, 4000, 100, cfg.count);
  countSlider.title = 'Applies on release: changing the body count regenerates the scenario.';
  countSlider.addEventListener('input', () => {
    countOut.textContent = Number(countSlider.value).toLocaleString();
  });
  countSlider.addEventListener('change', () => cfg.onCountChange(Number(countSlider.value)));

  const dtOut = document.createElement('output');
  const dtSlider = slider(0.0005, 0.02, 0.0005, cfg.dt);
  dtSlider.title =
    'Applies live. Changing dt mid-run breaks leapfrog’s symplectic energy ' +
    'conservation — a small one-off energy shift is expected, not a bug.';
  dtSlider.addEventListener('input', () => {
    const dt = Number(dtSlider.value);
    dtOut.textContent = dt.toFixed(4);
    cfg.onDtChange(dt);
  });

  root.append(
    row('', playBtn, resetBtn),
    row('Scenario', select),
    row('Bodies', countSlider, countOut),
    row('Timestep', dtSlider, dtOut),
  );

  const syncScenario = (scenario: Scenario, count: number, dt: number): void => {
    countSlider.disabled = !scenario.supportsN;
    if (scenario.supportsN) countSlider.value = String(count);
    countOut.textContent = count.toLocaleString();
    dtSlider.value = String(dt);
    dtOut.textContent = dt.toFixed(4);
  };
  syncScenario(
    cfg.scenarios.find((sc) => sc.id === cfg.activeId) ?? cfg.scenarios[0],
    cfg.count,
    cfg.dt,
  );

  return { syncScenario };
}
