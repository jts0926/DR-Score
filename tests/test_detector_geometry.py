import torch

from drscore.detector import square_crop_box


def test_square_crop_uses_horizontal_span_and_margin():
    crop = square_crop_box(torch.tensor([100, 80, 300, 260]), (500, 400), margin_pixels=10)
    assert crop == (90, 60, 310, 280)
    assert crop[2] - crop[0] == crop[3] - crop[1]
