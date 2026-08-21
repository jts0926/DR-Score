import numpy as np

from drscore.preprocessing import DRScorePreprocessor, rescale_dr_score


def test_preprocessing_shape_and_standardisation():
    image = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    output = DRScorePreprocessor(image_size=70)(image)
    assert output.shape == (1, 70, 70)
    assert abs(float(output.mean())) < 0.02
    assert 0.95 < float(output.std(unbiased=False)) < 1.05


def test_fixed_score_scaling():
    assert rescale_dr_score(-4.236161, -4.236161, 3.5887303) == 0.0
    assert rescale_dr_score(3.5887303, -4.236161, 3.5887303) == 4.0
