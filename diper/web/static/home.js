// Home page: hero, facts from the repo, and links to the pages.
import { route, app, api, esc, isStatic } from './core.js';

route('home', async () => {
  const info = await api('/info');
  const d = info.defaults;
  app().innerHTML = `<section class="hero"><p class="kicker">Cell migration · directional persistence</p>
    <h1 class="glow">DiPer</h1>
    <p class="lead">${esc(info.description)} Load your own track CSVs (one file per condition), give the frame interval and the
      pixel size, and get speed, directionality ratio, MSD, direction and velocity autocorrelation with error bars and CSV exports.</p>
    <a class="btn" href="#/analyse/demo">Try it on simulated tracks</a><a class="btn alt" href="#/analyse">Analyse your files</a></section>
    <div class="tiles">
      <div class="tile"><b>${info.analyses.length}</b><span>DiPer programs in the browser</span></div>
      <div class="tile"><b>${d.max_intervals}</b><span>autocorrelation intervals by default</span></div>
      <div class="tile"><b>N/2</b><span>longest MSD interval (half the track)</span></div>
      <div class="tile"><b>${isStatic() ? 'JS' : 'Python'}</b><span>engine on this ${isStatic() ? 'site (nothing uploaded)' : 'app'}</span></div></div>
    <div class="callout small">Direction autocorrelation, the measure DiPer was written for, averages cos θ between movement
      directions a time interval apart. Unlike the directionality ratio d/D it does not depend on how fast the cells move, so
      conditions with different speeds can be compared.</div>
    <div class="grid">
      <a class="card" href="#/analyse"><b>Analyse</b><span>Upload CSVs or use the simulated demo: tracks, tracks at the origin, speed, Dir_Ratio, MSD with α, Autocorrel, Vel_Cor, summary tables and CSV export.</span></a>
      <a class="card" href="#/methods"><b>Methods</b><span>What each program computes, in the repo's own conventions (time intervals, averaging, error bars, Sparse_Data).</span></a>
      <a class="card" href="#/format"><b>Data format</b><span>The DiPer column layout (frame, x, y in columns 4-6), how tracks and conditions are recognised, units.</span></a>
      <a class="card" href="#/about"><b>About</b><span>The original Excel macros (Gorelik &amp; Gautreau 2014), running the Python package and this web app.</span></a>
    </div>`;
});
