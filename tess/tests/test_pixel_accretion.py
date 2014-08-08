#!/usr/bin/env python
# encoding: utf-8
"""
Tests for the pixel_accretion module
"""
import numpy as np
from astropy.io import fits

from tess.pixel_accretion import IsoIntensityAccretor
from tess.pixel_accretion import EqualSNAccretor


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
    print accretor._seg_image
    # Should only be 4 groups
    assert accretor._seg_image.max() == 3


def test_iso_sn_image():
    """Make an image where each pixel has S/N=5."""
    img = 5. * np.ones((64, 64), dtype=float)
    noise = np.ones((64, 64), dtype=float)
    accretor = EqualSNAccretor(img, noise, 20.)
    fits.writeto("iso_sn.fits", accretor._seg_image, clobber=True)
    # Should be about 1024 groups.
    assert accretor._seg_image.max() <= 1050.
