const ORDER_STEP = 20;
const ORDER_LEVELS = 9;
const TIME_SPAN = 180;
const MAX_STROKES = 200;
const MAX_POINTS = 20000;

function total(values) {
  let sum = 0;
  let carry = 0;
  for (const value of values) {
    const next = sum + value;
    carry += Math.abs(sum) >= Math.abs(value) ? sum - next + value : value - next + sum;
    sum = next;
  }
  return carry ? sum + carry : sum;
}

export function simplify(points, epsilon = 2) {
  const keep = new Array(points.length).fill(false);
  keep[0] = keep[points.length - 1] = true;
  const stack = [[0, points.length - 1]];
  while (stack.length) {
    const [first, last] = stack.pop();
    const [x0, y0] = points[first];
    const [x1, y1] = points[last];
    const dx = x1 - x0;
    const dy = y1 - y0;
    const length = Math.sqrt(dx * dx + dy * dy);
    let worst = 0;
    let index = -1;
    for (let i = first + 1; i < last; i++) {
      const [px, py] = points[i];
      const distance = length === 0
        ? Math.sqrt((px - x0) ** 2 + (py - y0) ** 2)
        : Math.abs(dy * (px - x0) - dx * (py - y0)) / length;
      if (distance > worst) {
        worst = distance;
        index = i;
      }
    }
    if (index >= 0 && worst > epsilon) {
      keep[index] = true;
      stack.push([first, index], [index, last]);
    }
  }
  return points.filter((_, i) => keep[i]);
}

export function normalize(strokes) {
  strokes = strokes.filter(s => s.length === 2 && s[0].length > 0 && s[0].length === s[1].length);
  const xs = strokes.flatMap(s => s[0]);
  const ys = strokes.flatMap(s => s[1]);
  if (!strokes.length || strokes.length > MAX_STROKES || xs.length > MAX_POINTS) return null;
  if (!xs.concat(ys).every(value => Number.isFinite(value) && Math.abs(value) < 1e7)) return null;
  const left = Math.min(...xs);
  const top = Math.min(...ys);
  const scale = 255 / Math.max(Math.max(...xs) - left, Math.max(...ys) - top, 1e-6);
  return strokes.map(([sx, sy]) => {
    let points = sx.map((x, i) => [Math.round((x - left) * scale), Math.round((sy[i] - top) * scale)]);
    points = points.filter((p, i) => i === 0 || p[0] !== points[i - 1][0] || p[1] !== points[i - 1][1]);
    points = simplify(points);
    return [points.map(p => p[0]), points.map(p => p[1])];
  });
}

export function render(strokes, size = 32, width = 2, pad = 1) {
  const area = size * size;
  const image = new Uint8Array(3 * area);
  strokes = strokes.filter(([xs]) => xs.length > 0);
  if (!strokes.length) return image;
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const [xs, ys] of strokes) {
    for (const x of xs) {
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
    }
    for (const y of ys) {
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
  }
  const span = Math.max(maxX - minX, maxY - minY, 1e-6);
  const k = (size - 2 * pad - width) / span;
  const ox = size / 2 - (minX + maxX) / 2 * k;
  const oy = size / 2 - (minY + maxY) / 2 * k;
  const paths = [];
  let length = 0;
  for (const [xs, ys] of strokes) {
    let px = xs.map(x => x * k + ox);
    let py = ys.map(y => y * k + oy);
    if (px.length === 1) {
      px = [px[0], px[0]];
      py = [py[0], py[0]];
    }
    paths.push([px, py]);
    length += total(px.slice(1).map((x, i) => Math.hypot(x - px[i], py[i + 1] - py[i])));
  }
  length = Math.max(length, 1e-6);
  const reach = width / 2 + 0.5;
  let walked = 0;
  paths.forEach(([px, py], index) => {
    const level = 255 - ORDER_STEP * Math.min(index, ORDER_LEVELS);
    for (let n = 0; n < px.length - 1; n++) {
      const x0 = px[n], y0 = py[n], x1 = px[n + 1], y1 = py[n + 1];
      const dx = x1 - x0;
      const dy = y1 - y0;
      const ll = dx * dx + dy * dy;
      const seg = Math.sqrt(ll);
      const t0 = walked / length;
      const dt = seg / length;
      walked += seg;
      const i0 = Math.max(0, Math.trunc(Math.min(x0, x1) - reach));
      const i1 = Math.min(size - 1, Math.trunc(Math.max(x0, x1) + reach));
      const j0 = Math.max(0, Math.trunc(Math.min(y0, y1) - reach));
      const j1 = Math.min(size - 1, Math.trunc(Math.max(y0, y1) + reach));
      const inv = ll > 0 ? 1 / ll : 0;
      for (let j = j0; j <= j1; j++) {
        const cy = j + 0.5 - y0;
        const row = j * size;
        for (let i = i0; i <= i1; i++) {
          const cx = i + 0.5 - x0;
          let s = (cx * dx + cy * dy) * inv;
          if (s < 0) s = 0;
          else if (s > 1) s = 1;
          const ex = s * dx - cx;
          const ey = s * dy - cy;
          let c = reach - Math.sqrt(ex * ex + ey * ey);
          if (c <= 0) continue;
          if (c > 1) c = 1;
          const p = row + i;
          let v = Math.trunc(255 * c + 0.5);
          if (v > image[p]) image[p] = v;
          v = Math.trunc(level * c + 0.5);
          if (v > image[area + p]) image[area + p] = v;
          v = Math.trunc((255 - TIME_SPAN * (t0 + s * dt)) * c + 0.5);
          if (v > image[2 * area + p]) image[2 * area + p] = v;
        }
      }
    }
  });
  return image;
}

export function softmax(logits) {
  let peak = -Infinity;
  for (const value of logits) if (value > peak) peak = value;
  const scores = Array.from(logits, value => Math.exp(value - peak));
  const sum = scores.reduce((a, b) => a + b, 0);
  return scores.map(value => value / sum);
}

export function read(output, meta, pixels, guesses = 5) {
  const scores = softmax(output.logits.data);
  const order = scores.map((_, i) => i).sort((a, b) => scores[b] - scores[a]).slice(0, guesses);
  const grid = output.maps.dims[2];
  const cells = Array.from(output.maps.data.slice(order[0] * grid * grid, (order[0] + 1) * grid * grid), v => Math.max(v, 0));
  const peak = Math.max(Math.max(...cells), 1e-9);
  const heat = Array.from({ length: grid }, (_, y) => cells.slice(y * grid, (y + 1) * grid).map(v => v / peak));
  const size = meta.size;
  const image = Array.from({ length: size }, (_, y) => Array.from(pixels.slice(y * size, (y + 1) * size)));
  return { guesses: order.map(i => ({ label: meta.classes[i], p: scores[i] })), heat, image };
}

export function tensor(pixels, meta) {
  const input = new Float32Array(meta.channels * meta.size * meta.size);
  for (let i = 0; i < input.length; i++) input[i] = pixels[i] / 255;
  return input;
}
