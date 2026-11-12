// About page: what the package is, how to run it and the web app, credits (from the README and the package docstring).
import { route, app, api, esc } from './core.js';

route('about', async () => {
  const info = await api('/info');
  app().innerHTML = `<p class="kicker">About</p><h1>DiPer in Python</h1>
    <p class="lead">A Python reimplementation (version ${esc(info.version)}) of DiPer, the Excel VBA macros published with
      "Quantitative and unbiased analysis of directional persistence in cell migration" (Gorelik &amp; Gautreau, Nature
      Protocols, 2014), for analysing cell migration trajectories.</p>
    <p class="small">Reference: <a href="${esc(info.reference.url)}" target="_blank" rel="noopener">${esc(info.reference.text)}</a></p>
    <h2>Use the Python package</h2>
    <pre>pip install -r requirements.txt
python - &lt;&lt;'EOF'
from diper.utils import load_data
from diper.msd import run_msd_analysis
from diper.autocorrel import run_autocorrelation_analysis
data = load_data("trajectories.xlsx")          # one sheet per condition
run_msd_analysis(data, time_interval=1.0, output_dir="results")
run_autocorrelation_analysis(data, time_interval=1.0, max_intervals=30, output_dir="results")
EOF</pre>
    <p>Each <code>run_*</code> function writes CSV tables and PNG/PDF plots to the output folder; <code>docs/usage_example.py</code>
      runs a whole workflow and <code>docs/diper_user_manual.md</code> describes every program. There is no command-line tool yet.</p>
    <h2>Run the web app</h2>
    <pre>pip install -r requirements-web.txt
python run_web.py        # http://127.0.0.1:8156</pre>
    <p>The local web app sends your files to its own Python server, which runs the package's functions
      (<code>diper/web/analysis.py</code>) and also accepts Excel. The static GitHub Pages site runs a JavaScript port of the same
      steps in your browser (<code>diper_engine.js</code>), checked against the Python results by a test to about 1e-15; nothing is
      uploaded. The demo tracks on both are simulated (seeded persistent random walks): the repository contains no measured data.</p>
    <h2>Credits</h2>
    <p>Method and original macros: Roman Gorelik &amp; Alexis Gautreau (2014). Built with AI assistance from
      <a href="https://claude.com/" target="_blank" rel="noopener">Claude (Anthropic)</a>. Licence: MIT.
      Source: <a href="https://github.com/gddickinson/${esc(info.repo)}" target="_blank" rel="noopener">github.com/gddickinson/${esc(info.repo)}</a>.</p>`;
});
