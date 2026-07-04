import type { ForceMethod, Scenario } from '../scenarios/types';

/** Slider bound for scenarios that don't declare their own maxN. */
const DEFAULT_MAX_N = 4000;

export interface ControlsConfig {
  scenarios: readonly Scenario[];
  activeId: string;
  count: number;
  dt: number;
  forceMethod: ForceMethod;
  theta: number;
  /** Toggle play/pause; returns the new running state. */
  onTogglePlay(): boolean;
  onReset(): void;
  onScenarioChange(id: string): void;
  /** Fires on slider release — a new count regenerates the scenario. */
  onCountChange(n: number): void;
  /** Fires live while dragging — dt applies to the running sim. */
  onDtChange(dt: number): void;
  /** Fires on select — applies to the running sim from the next step. */
  onForceMethodChange(method: ForceMethod): void;
  /** Fires live while dragging — θ applies from the next step. */
  onThetaChange(theta: number): void;
}

export interface Controls {
  /**
   * Sync widgets after a scenario switch (count bounds, dt, force method,
   * enablement).
   */
  syncScenario(scenario: Scenario, count: number, dt: number, forceMethod: ForceMethod): void;
  /**
   * Enable the GPU option (WebGPU detection succeeded). The option starts
   * disabled with a "checking" note until one of these two is called.
   */
  enableGpu(): void;
  /**
   * Disable the GPU option (no adapter / device lost). If GPU was the
   * live selection, the change is routed through the normal change event
   * so the app's method state and θ enablement stay in sync without any
   * caller-side compensation.
   */
  disableGpu(note: string): void;
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
  const countSlider = slider(100, DEFAULT_MAX_N, 100, cfg.count);
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

  const methodSelect = document.createElement('select');
  const methodOption = (value: ForceMethod, label: string): HTMLOptionElement => {
    const option = document.createElement('option');
    option.value = value;
    option.textContent = label;
    methodSelect.append(option);
    return option;
  };
  methodOption('brute', 'Brute force (exact)');
  methodOption('barnes-hut', 'Barnes–Hut (θ)');
  const gpuOption = methodOption('gpu', 'GPU brute force (f32)');
  methodSelect.title =
    'Brute force is exact O(N²); Barnes–Hut approximates far-field forces ' +
    'via a quadtree, O(N log N); GPU runs the exact O(N²) sum in Float32 ' +
    'compute shaders (watch ΔE/E₀ for what the precision drop costs).';

  const gpuNote = document.createElement('div');
  gpuNote.className = 'note';
  const enableGpu = (): void => {
    gpuOption.disabled = false;
    gpuNote.hidden = true;
  };
  const disableGpu = (note: string): void => {
    gpuOption.disabled = true;
    gpuNote.textContent = note;
    gpuNote.hidden = false;
    if (methodSelect.value === 'gpu') {
      methodSelect.value = 'barnes-hut';
      // Route through the normal change path: onForceMethodChange and
      // syncTheta must observe this like any user-driven switch.
      methodSelect.dispatchEvent(new Event('change'));
    }
  };
  // Detection is async (see main.ts); until it lands the option is
  // disabled with an honest placeholder.
  disableGpu('Checking WebGPU support…');

  const thetaOut = document.createElement('output');
  const thetaSlider = slider(0, 1, 0.05, cfg.theta);
  thetaSlider.title =
    'Barnes–Hut opening angle. Larger θ accepts coarser far-field ' +
    'approximations: faster, less accurate (watch ΔE/E₀ in the overlay). ' +
    'θ = 0 degenerates to exact brute force.';
  const syncTheta = (): void => {
    thetaOut.textContent = Number(thetaSlider.value).toFixed(2);
    thetaSlider.disabled = methodSelect.value !== 'barnes-hut';
  };
  thetaSlider.addEventListener('input', () => {
    syncTheta();
    cfg.onThetaChange(Number(thetaSlider.value));
  });
  methodSelect.value = cfg.forceMethod;
  methodSelect.addEventListener('change', () => {
    syncTheta();
    cfg.onForceMethodChange(methodSelect.value as ForceMethod);
  });

  root.append(
    row('', playBtn, resetBtn),
    row('Scenario', select),
    row('Bodies', countSlider, countOut),
    row('Timestep', dtSlider, dtOut),
    row('Forces', methodSelect),
    gpuNote,
    row('θ', thetaSlider, thetaOut),
  );

  const syncScenario = (
    scenario: Scenario,
    count: number,
    dt: number,
    forceMethod: ForceMethod,
  ): void => {
    countSlider.disabled = !scenario.supportsN;
    countSlider.max = String(scenario.maxN ?? DEFAULT_MAX_N);
    if (scenario.supportsN) countSlider.value = String(count);
    countOut.textContent = count.toLocaleString();
    dtSlider.value = String(dt);
    dtOut.textContent = dt.toFixed(4);
    methodSelect.value = forceMethod;
    syncTheta();
  };
  syncScenario(
    cfg.scenarios.find((sc) => sc.id === cfg.activeId) ?? cfg.scenarios[0],
    cfg.count,
    cfg.dt,
    cfg.forceMethod,
  );

  return { syncScenario, enableGpu, disableGpu };
}
