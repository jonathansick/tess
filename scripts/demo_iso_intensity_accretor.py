#!/usr/bin/env python
# encoding: utf-8
"""
Demonstration of the equal-intensity accretion recipe on a input image
with quandrants of different intensities.
"""

import numpy as np
from astropy.io import fits

from tess.pixel_accretion import IsoIntensityAccretor


def main():
    img = np.zeros((64, 64), dtype=float)
    img[0:32, 0:32] = 1.
    img[0:32, 32:] = -10.
    img[32:, 0:32] = 100.
    img[32:, 32:] = 1000.
    accretor = IsoIntensityAccretor(img, 0.1)
    accretor.accrete((0, 0))
    print accretor._seg_image
    fits.writeto("iso_intensity.fits", accretor._seg_image, clobber=True)


if __name__ == '__main__':
    main()
