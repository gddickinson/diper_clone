// CSV reading for the JS port, mirroring diper.utils.load_data (CSV branch) + the web layer's preparation
// (diper/web/analysis.py: prepare): the first row is a header; one CSV file is one condition, named after the file;
// columns are taken BY POSITION: 4 = frame, 5 = x, 6 = y, 7 = z (optional); columns 1-3 are free metadata.

// Strings pandas.read_csv reads as missing (its default na_values).
const NA = new Set(['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN', '<NA>', 'N/A',
  'NA', 'NULL', 'NaN', 'None', 'n/a', 'nan', 'null']);
const NUM = /^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/;

// pd.to_numeric(errors='coerce') for one cell: NaN for missing or non-numeric text.
export function toNum(v) {
  const t = String(v ?? '').trim();
  if (NA.has(t) || !NUM.test(t)) return NaN;
  return Number(t);
}

// Minimal RFC-4180 CSV reader: quoted fields, "" escapes, CRLF; blank lines skipped (as pandas does).
export function parseCSV(text) {
  const rows = [];
  let row = [], field = '', q = false;
  const s = String(text).replace(/^﻿/, '');
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if (q) {
      if (c === '"') { if (s[i + 1] === '"') { field += '"'; i++; } else q = false; } else field += c;
    } else if (c === '"') q = true;
    else if (c === ',') { row.push(field); field = ''; }
    else if (c === '\n' || c === '\r') {
      if (c === '\r' && s[i + 1] === '\n') i++;
      row.push(field); field = '';
      if (row.length > 1 || row[0] !== '') rows.push(row);
      row = [];
    } else field += c;
  }
  row.push(field);
  if (row.length > 1 || row[0] !== '') rows.push(row);
  if (!rows.length) throw new Error('No columns to parse from file');
  return { header: rows[0], rows: rows.slice(1) };
}

// Condition name as load_data gives it: the file name without extension, after the web layer's _safe_name.
export function conditionName(filename) {
  const base = String(filename || 'upload.csv').split(/[\\/]/).pop();
  const dot = base.lastIndexOf('.');
  const stem = (dot > 0 ? base.slice(0, dot) : base).replace(/[^\p{L}\p{N}_.\- ()]+/gu, '_').trim();
  return stem || 'upload';
}

// One condition table from CSV text: { name, n_rows, frame, x, y, z|null } with raw numbers (NaN where missing).
export function readCondition(text, filename) {
  const { header, rows } = parseCSV(text);
  if (header.length < 6) throw new Error("CSV file doesn't have enough columns.");
  const col = k => rows.map(r => toNum(r[k]));
  return { name: conditionName(filename), n_rows: rows.length, frame: col(3), x: col(4), y: col(5),
    z: header.length >= 7 ? col(6) : null };
}
