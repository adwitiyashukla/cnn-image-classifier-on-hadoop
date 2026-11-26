import base64
import math
import zlib

ORDER_STEP = 20
ORDER_LEVELS = 9
TIME_SPAN = 180


def render(strokes, size=32, width=2.0, pad=1.0):
    area = size * size
    ink = bytearray(area)
    order = bytearray(area)
    clock = bytearray(area)
    strokes = [(xs, ys) for xs, ys in strokes if xs]
    if not strokes:
        return bytes(3 * area)
    xs_all = [x for xs, _ in strokes for x in xs]
    ys_all = [y for _, ys in strokes for y in ys]
    min_x, max_x, min_y, max_y = min(xs_all), max(xs_all), min(ys_all), max(ys_all)
    span = max(max_x - min_x, max_y - min_y, 1e-6)
    k = (size - 2 * pad - width) / span
    ox = size / 2 - (min_x + max_x) / 2 * k
    oy = size / 2 - (min_y + max_y) / 2 * k
    paths = []
    total = 0.0
    for xs, ys in strokes:
        px = [x * k + ox for x in xs]
        py = [y * k + oy for y in ys]
        if len(px) == 1:
            px, py = px * 2, py * 2
        paths.append((px, py))
        total += sum(math.hypot(px[i + 1] - px[i], py[i + 1] - py[i]) for i in range(len(px) - 1))
    total = max(total, 1e-6)
    reach = width / 2 + 0.5
    sqrt = math.sqrt
    walked = 0.0
    for index, (px, py) in enumerate(paths):
        level = 255 - ORDER_STEP * min(index, ORDER_LEVELS)
        for n in range(len(px) - 1):
            x0, y0, x1, y1 = px[n], py[n], px[n + 1], py[n + 1]
            dx, dy = x1 - x0, y1 - y0
            ll = dx * dx + dy * dy
            seg = sqrt(ll)
            t0 = walked / total
            dt = seg / total
            walked += seg
            i0 = max(0, int(min(x0, x1) - reach))
            i1 = min(size - 1, int(max(x0, x1) + reach))
            j0 = max(0, int(min(y0, y1) - reach))
            j1 = min(size - 1, int(max(y0, y1) + reach))
            inv = 1.0 / ll if ll > 0 else 0.0
            for j in range(j0, j1 + 1):
                cy = j + 0.5 - y0
                row = j * size
                for i in range(i0, i1 + 1):
                    cx = i + 0.5 - x0
                    s = (cx * dx + cy * dy) * inv
                    if s < 0.0:
                        s = 0.0
                    elif s > 1.0:
                        s = 1.0
                    ex = s * dx - cx
                    ey = s * dy - cy
                    c = reach - sqrt(ex * ex + ey * ey)
                    if c <= 0.0:
                        continue
                    if c > 1.0:
                        c = 1.0
                    p = row + i
                    v = int(255 * c + 0.5)
                    if v > ink[p]:
                        ink[p] = v
                    v = int(level * c + 0.5)
                    if v > order[p]:
                        order[p] = v
                    v = int((255 - TIME_SPAN * (t0 + s * dt)) * c + 0.5)
                    if v > clock[p]:
                        clock[p] = v
    return bytes(ink + order + clock)


def encode(image):
    return base64.b64encode(zlib.compress(image, 6)).decode("ascii")


def decode(text):
    return zlib.decompress(base64.b64decode(text))
