#!/usr/bin/env python
# encoding: utf-8
"""
Test the starfield stellar density rendering with *real* data from
the Hubble Legacy Archive.

Dataset: HST_10584_18_ACS_WFC_F606W_F435W

2012-09-24 - Created by Jonathan Sick
"""

import numpy as np
import pyfits

from ghostmap.starfield import StarField


def main():
    daoCatPath = "HST_10584_18_ACS_WFC_multiwave_daophot_trm.cat"
    headerPath = "color_HST_10584_18_ACS_WFC_F606W_F435W_sci_head.txt"

    header = pyfits.core.Header(txtfile=headerPath)

    x, y, f435w, f606w, flag1, flag2 = np.loadtxt(daoCatPath,
            usecols=(1, 2, 7, 8, 11, 12), unpack=True)
    flags = flag1 + flag2
    good = np.where((flags < 1) & (f435w < 30.) & (f606w < 30.))[0]
    x = x[good]
    y = y[good]
    f435w = f435w[good]
    f606w = f606w[good]
    weight = np.ones(len(f606w), dtype=np.float)

    starField = StarField.load_arrays(x, y, f435w - f606w, f606w,
            weight=weight, header=header)
    starField.select_colours([(-1, 20.), (-1, 26.), (3, 26), (3, 20.)])
    starField.plot_colour_selection("test_hla_cmd_sel",
            xLabel="F435W - F606W", yLabel="F606W")
    starField.estimate_density_field(10.)
    starField.save_fits("hla_density_field.fits")


if __name__ == '__main__':
    main()
