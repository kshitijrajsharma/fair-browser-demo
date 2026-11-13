import numpy as np
from PIL import Image

from compare import HOT_MEAN, HOT_STD, iou, load_tile


def test_iou_of_identical_masks_is_one():
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True
    assert iou(mask, mask) == 1.0


def test_iou_of_disjoint_masks_is_zero():
    first = np.zeros((4, 4), dtype=bool)
    second = np.zeros((4, 4), dtype=bool)
    first[0, 0] = True
    second[3, 3] = True
    assert iou(first, second) == 0.0


def test_iou_of_two_empty_masks_is_one():
    empty = np.zeros((4, 4), dtype=bool)
    assert iou(empty, empty) == 1.0


def test_load_tile_returns_normalised_chw_batch(tmp_path):
    path = tmp_path / "tile.png"
    Image.new("RGBA", (256, 256), (255, 0, 128, 255)).save(path)
    tile = load_tile(path)
    assert tile.shape == (1, 3, 256, 256)
    expected = (np.array([255, 0, 128]) / 255.0 - HOT_MEAN) / HOT_STD
    np.testing.assert_allclose(tile[0, :, 0, 0], expected, rtol=1e-5)
