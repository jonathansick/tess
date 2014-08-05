#!/usr/bin/env python
# encoding: utf-8
"""
Demonstration of the equal-SN accretion recipe.

2014-08-01 - Created by Jonathan Sick
"""

import numpy as np
from astropy.io import fits

# import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.gridspec as gridspec

from tess.pixel_accretion import EqualSNAccretor


def main():
    img = 5. * np.ones((64, 64), dtype=float)
    noise = np.ones((64, 64), dtype=float)
    accretor = EqualSNAccretor(img, noise, 20.)
    accretor.accrete((0, 0))
    accretor.cleanup()
    fits.writeto("iso_sn.fits", accretor.segimage, clobber=True)

    segimage = accretor.segimage
    centroids = accretor.centroids
    x = np.array([c[1][0] for c in centroids])
    y = np.array([c[1][1] for c in centroids])

    fig = Figure(figsize=(3.5, 3.5), frameon=False)
    canvas = FigureCanvas(fig)
    gs = gridspec.GridSpec(1, 1,
                           left=0.15, right=0.95, bottom=0.15, top=0.95,
                           wspace=None, hspace=None,
                           width_ratios=None, height_ratios=None)
    ax = fig.add_subplot(gs[0])
    ax.imshow(segimage, interpolation='None', aspect='equal')
    ax.scatter(x, y, c='k', marker='o', s=4)
    gs.tight_layout(fig, pad=1.08, h_pad=None, w_pad=None, rect=None)
    canvas.print_figure("iso_sn.png", format="png")


if __name__ == '__main__':
    main()
