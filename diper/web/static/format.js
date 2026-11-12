// Data format page: what diper.utils.load_data reads, with an example and the demo download.
import { route, app, isStatic, download } from './core.js';
import { simulateFiles } from './diper_sim.js';

route('format', async () => {
  app().innerHTML = `<p class="kicker">Input</p><h1>Data format</h1>
    <p class="lead">The DiPer layout: one table of positions per condition, one row per frame of each track, tracks one after
      another. CSV works everywhere (one file = one condition, named after the file); Excel (.xlsx/.xls, one sheet = one
      condition) works in the local app${isStatic() ? ' only' : ''}.</p>
    <div class="panel"><h3 style="margin-top:0">Columns (by position, the header names do not matter)</h3>
      <table><tr><th>Column</th><th>Content</th></tr>
        <tr><td>1-3</td><td>free metadata (cell, experiment, condition...), ignored</td></tr>
        <tr><td>4</td><td>frame number</td></tr>
        <tr><td>5</td><td>x</td></tr>
        <tr><td>6</td><td>y</td></tr>
        <tr><td>7 (optional)</td><td>z, used by Autocorrel_3D</td></tr></table>
      <p class="small">The first row is a header. A table with fewer than 6 columns is rejected
        ("CSV file doesn't have enough columns."). Columns after the 7th are ignored.</p></div>
    <div class="panel"><h3 style="margin-top:0">How tracks are recognised</h3>
      <p><b>A new track starts wherever the frame number does not increase</b> from one row to the next (it drops or repeats).
        So list each track's rows in frame order and start the next track at a frame number no higher than the last one.
        Gaps in frame numbers are not filled: consecutive rows count as consecutive frames, Δt apart.</p>
      <p>Rows with a missing or non-numeric frame, x or y are dropped (the web pages report how many). One-point tracks are dropped.</p></div>
    <div class="panel"><h3 style="margin-top:0">Units</h3>
      <p>Enter whether x/y are in pixels (then the pixel size in µm per pixel is required; it is never assumed) or already in
        micrometres. The web pages convert to µm before analysing, so speed is in µm per time unit and MSD in µm²; d/D,
        α and the autocorrelations have no unit. A z column must use the same unit as x and y (it is scaled by the same pixel
        size). The time interval is the time between frames, in the unit you choose.</p></div>
    <h2>Example</h2>
    <pre>Cell,Experiment,Condition,Frame,X,Y
cell1,exp1,control,1,212.40,180.02
cell1,exp1,control,2,214.97,181.33
cell1,exp1,control,3,217.10,183.05
cell2,exp1,control,1,64.81,300.55      &lt;- frame drops: a new track
cell2,exp1,control,2,63.02,302.90</pre>
    <p><button id="dl" type="button">Download the simulated demo CSVs</button>
      <span class="small mute">Seeded persistent random walks in exactly this layout, x/y in µm. Not measured data.</span></p>`;
  document.getElementById('dl').onclick = () => simulateFiles().forEach(f => download(f.name, f.text, 'text/csv'));
});
