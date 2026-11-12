// Static-site data layer: answers api() calls without the Python server.
// GET paths come from the JSON exported by scripts/build_static.py; POST /analyze runs the JS port (diper_engine.js),
// which is tested against the repo's Python (diper_engine.test.mjs).
import { analyseFiles } from './diper_engine.js';

const files = new Map();
function file(rel) {
  if (!files.has(rel)) files.set(rel, fetch('api/' + rel + '.json').then(r => { if (!r.ok) throw new Error(`${r.status} ${rel}`); return r.json(); }));
  return files.get(rel);
}

export async function staticApi(path, body) {
  const clean = path.split('?')[0].replace(/^\/+/, '');
  if (clean === 'analyze') {
    const list = body ? body.getAll('files') : [];
    if (!list.length) throw new Error('No file given.');
    const texts = await Promise.all(list.map(async f => ({ name: f.name, text: /\.csv$/i.test(f.name) ? await f.text() : '' })));
    return analyseFiles(texts, body.get('time_interval'), body.get('max_intervals'), body.get('sparse_n'), body.get('um_per_px') || null, body.get('weighting') || 'displacements');
  }
  return file(clean);
}
