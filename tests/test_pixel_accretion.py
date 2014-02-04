#!/usr/bin/env python
# encoding: utf-8
"""
Tests for the pixel_accretion module
"""
import numpy as np

from tess.pixel_accretion import IsoIntensityAccretor


def test_isointensity_blockimage():
    """Make a block of 4 contrasting intensities and apply
    IsoIntensityPixelAccretor.
    """
    img = np.zeros((16, 16), dtype=float)
    img[0:8, 0:8] = 1.
    img[0:8, 8:16] = -10.
    img[8:16, 0:8] = 100.
    img[8:16, 8:16] = 1000.
    print "image"
    print img
    accretor = IsoIntensityAccretor(img, 0.1)
    accretor.accrete((0, 0))
    print accretor._seg_image
    # Should only be 4 groups
    assert accretor._seg_image.max() == 3


if __name__ == '__main__':
    test_isointensity_blockimage()
