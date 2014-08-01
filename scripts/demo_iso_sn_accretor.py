#!/usr/bin/env python
# encoding: utf-8
"""
Demonstration of the equal-SN accretion recipe.

2014-08-01 - Created by Jonathan Sick
"""

import numpy as np
from astropy.io import fits

from tess.pixel_accretion import EqualSNAccretor


def main():
    img = 5. * np.ones((64, 64), dtype=float)
    noise = np.ones((64, 64), dtype=float)
    accretor = EqualSNAccretor(img, noise, 20.)
    accretor.accrete((0, 0))
    fits.writeto("iso_sn.fits", accretor._seg_image, clobber=True)


if __name__ == '__main__':
    main()
