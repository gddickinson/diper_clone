// CSV exports of an analysis result. Each file starts with '#' comment lines recording the settings and units, so the
// numbers stay traceable (read back with pandas.read_csv(path, comment='#')).

const q = v => (/[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v));
const n = v => (v === null || v === undefined ? '' : String(v));

export function header(res, meta) {
  return [
    `# DiPer results (engine: ${res.engine === 'python' ? 'Python package diper (diper/web/analysis.py)' : 'JS port diper_engine.js'})`,
    `# files: ${res.files.join('; ')}${meta.simulated ? ' (SIMULATED demo data, not measurements)' : ''}`,
    `# time_interval: ${res.time_interval} ${meta.timeUnit} per frame; sparse_n: ${res.sparse_n}; time interval used: ${res.dt_eff} ${meta.timeUnit}`,
    `# coordinates: ${res.um_per_px ? `pixels x ${res.um_per_px} um/px` : 'micrometres'}; lengths in um, speed in um/${meta.timeUnit}, MSD in um^2`,
    `# autocorrelation intervals: ${res.max_intervals}; SEM = population std (ddof 0) / sqrt(n tracks), except d/D last point (ddof 1)`,
  ];
}

// One row per condition: the headline numbers.
export function summaryCSV(res, meta) {
  const lines = header(res, meta);
  lines.push(`condition,n_tracks,speed_mean_um_per_${meta.timeUnit},speed_sem,dir_ratio_last_mean,dir_ratio_last_sem,msd_alpha,msd_alpha_r2,msd_alpha_n_points`);
  for (const c of res.conditions) {
    if (c.error) { lines.push(`${q(c.name)},,,,,,,,# error: ${q(c.error)}`); continue; }
    lines.push([q(c.name), c.n_tracks, n(c.speed.mean), n(c.speed.sem), n(c.dir_ratio.last_mean), n(c.dir_ratio.last_sem),
      n(c.msd.alpha), n(c.msd.r2), c.msd.n_fit].join(','));
  }
  return lines.join('\n') + '\n';
}

// Long format: every averaged curve (Dir_Ratio, MSD, Autocorrel, Autocorrel_3D, Vel_Cor) with mean, SEM and n tracks.
export function curvesCSV(res, meta) {
  const lines = header(res, meta);
  lines.push(`condition,analysis,time_${meta.timeUnit},mean,sem,n_tracks`);
  const keys = [['dir_ratio', 'dir_ratio'], ['msd', 'msd_um2'], ['autocorr', 'autocorrel'], ['autocorr3d', 'autocorrel_3d'], ['vel_cor', 'vel_cor']];
  for (const c of res.conditions) {
    if (c.error) continue;
    for (const [k, label] of keys) {
      const cv = c[k];
      if (cv) cv.t.forEach((t, i) => lines.push([q(c.name), label, t, n(cv.avg[i]), n(cv.sem[i]), cv.n[i]].join(',')));
    }
  }
  return lines.join('\n') + '\n';
}

// Per-cell values: average speed and last-point directionality ratio (as cell_speeds_* / dir_ratio_last_* files).
export function cellsCSV(res, meta) {
  const lines = header(res, meta);
  lines.push(`condition,measure,cell_number,value`);
  for (const c of res.conditions) {
    if (c.error) continue;
    c.speed.cells.forEach((v, i) => lines.push([q(c.name), `average_speed_um_per_${meta.timeUnit}`, i + 1, n(v)].join(',')));
    c.dir_ratio.last.forEach((v, i) => lines.push([q(c.name), 'dir_ratio_last', i + 1, n(v)].join(',')));
  }
  return lines.join('\n') + '\n';
}
