// Numeric helpers that follow numpy/pandas summation order, so the JS port agrees with Python to ~1e-15.
// numpy's add.reduce is a pairwise sum (sequential below 8 elements, 8 accumulators up to 128, halves above; checked
// bit for bit against numpy 2.5); np.mean = sum / n, np.std = sqrt(sum((x - mean)^2) / n) (ddof 0); pandas Series.mean/.sem use the same
// sums (sem with ddof 1) and skip NaN.

function pairwise(a, lo, n) {
  if (n < 8) {
    let res = -0.0;
    for (let i = 0; i < n; i++) res += a[lo + i];
    return res;
  }
  if (n <= 128) {
    const r = [];
    for (let j = 0; j < 8; j++) r[j] = a[lo + j];
    let i = 8;
    for (; i < n - (n % 8); i += 8) for (let j = 0; j < 8; j++) r[j] += a[lo + i + j];
    let res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; i++) res += a[lo + i];
    return res;
  }
  let n2 = Math.trunc(n / 2);
  n2 -= n2 % 8;
  return pairwise(a, lo, n2) + pairwise(a, lo + n2, n - n2);
}

// np.sum / np.add.reduce of a 1-D float array.
export function npSum(a) {
  return a.length ? pairwise(a, 0, a.length) : 0;
}

export const npMean = a => (a.length ? npSum(a) / a.length : NaN);

export function npStd(a) {
  const m = npSum(a) / a.length;
  return Math.sqrt(npSum(a.map(v => { const d = v - m; return d * d; })) / a.length);
}

// SEM as the repo computes it: np.std (ddof 0) / sqrt(n), NaN for a single value.
export const semPop = a => (a.length > 1 ? npStd(a) / Math.sqrt(a.length) : NaN);

// pandas Series.mean() and .sem() (ddof 1), skipping NaN.
export function pdMean(a) {
  const v = a.filter(x => !Number.isNaN(x));
  return v.length ? npSum(v) / v.length : NaN;
}
export function pdSem(a) {
  const v = a.filter(x => !Number.isNaN(x)), n = v.length;
  if (n < 2) return NaN;
  const avg = npSum(v) / n;
  const variance = npSum(v.map(x => (avg - x) ** 2)) / (n - 1);
  return Math.sqrt(variance) / Math.sqrt(n);
}

// Least-squares straight line (np.polyfit(x, y, 1)) and R^2 as msd.calculate_alpha_values computes it.
export function lineFit(x, y) {
  const mx = npMean(x), my = npMean(y);
  let sxy = 0, sxx = 0;
  for (let i = 0; i < x.length; i++) { sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx) ** 2; }
  const slope = sxy / sxx, intercept = my - slope * mx;
  const pred = x.map(v => slope * v + intercept);
  const ssTot = npSum(y.map(v => (v - my) ** 2)), ssRes = npSum(y.map((v, i) => (v - pred[i]) ** 2));
  return { slope, intercept, r2: ssTot !== 0 ? 1 - ssRes / ssTot : NaN };
}
