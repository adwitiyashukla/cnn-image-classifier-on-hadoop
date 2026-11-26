import numpy as np

from imgcnn import raster


def planes(strokes, size=32, width=2.0):
    return np.frombuffer(raster.render(strokes, size, width), np.uint8).reshape(3, size, size)


def test_output_has_three_planes():
    image = raster.render([[[0, 255], [0, 255]]], 32, 2.0)
    assert len(image) == 3 * 32 * 32
    assert max(image) == 255


def test_scale_and_shift_do_not_change_the_image():
    base = [[[10, 60, 90, 20], [15, 80, 30, 70]], [[40, 45], [5, 95]]]
    moved = [[[x * 2.5 + 300 for x in xs], [y * 2.5 - 40 for y in ys]] for xs, ys in base]
    difference = np.abs(planes(base).astype(int) - planes(moved).astype(int))
    assert difference.max() <= 1


def test_drawing_is_centered_and_keeps_aspect_ratio():
    ink = planes([[[0, 255], [100, 100]]])[0]
    rows = np.nonzero(ink.sum(1))[0]
    columns = np.nonzero(ink.sum(0))[0]
    assert columns[0] <= 1 and columns[-1] >= 30
    assert 13 <= rows[0] and rows[-1] <= 18


def test_order_and_time_never_exceed_ink():
    ink, order, clock = planes([[[0, 200, 30], [0, 90, 255]], [[255, 10], [20, 240]], [[120, 130], [0, 255]]])
    assert (order <= ink).all()
    assert (clock <= ink).all()


def test_later_strokes_are_lighter_in_the_order_plane():
    order = planes([[[0, 255], [0, 0]], [[0, 255], [255, 255]]])[1]
    assert order[:8].max() == 255
    assert order[-8:].max() == 255 - raster.ORDER_STEP


def test_time_plane_fades_along_the_path():
    clock = planes([[[0, 255], [0, 0]]])[2]
    row = clock[clock.sum(1).argmax()]
    assert row[3] > row[16] > row[28]
    assert row[28] >= 255 - raster.TIME_SPAN - 2


def test_empty_and_single_point_drawings():
    assert raster.render([], 32, 2.0) == bytes(3 * 32 * 32)
    assert raster.render([[[], []]], 32, 2.0) == bytes(3 * 32 * 32)
    assert planes([[[7], [9]]])[0].max() > 150


def test_other_sizes():
    for size in (16, 48, 64):
        assert planes([[[0, 255, 0], [0, 255, 255]]], size, size / 16)[0].max() == 255


def test_encode_round_trip():
    image = raster.render([[[0, 255, 30], [0, 255, 200]]], 32, 2.0)
    text = raster.encode(image)
    assert text.isascii() and "\t" not in text and "\n" not in text
    assert raster.decode(text) == image
    assert raster.decode(text.encode()) == image
