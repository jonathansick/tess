#!/usr/bin/env python
# encoding: utf-8
"""
Equal-SN Centroidal Voronoi Tessellation
"""

import numpy as np

# import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.gridspec as gridspec

from tess.pixel_accretion import EqualSNAccretor
from tess.cvtessellation import CVTessellation


def main():
    shape = (256, 256)
    pix_x, pix_y = np.meshgrid(np.arange(shape[1], dtype=float),
                               np.arange(shape[0], dtype=float))
    # img = 5. * np.ones(shape, dtype=float)
    # noise = np.ones(shape, dtype=float)
    img = 10. * np.exp(-((pix_x - 128.) ** 2. / (2. * 50. ** 2.) +
                         (pix_y - 128.) ** 2. / (2. * 50. ** 2.)))
    # noise = 50. * np.sqrt(img)
    noise = np.ones(shape, dtype=float)
    accretor = EqualSNAccretor(img, noise, 50., start=(0, 0))
    accretor.cleanup()
    print "accretor.bin_sn", accretor.bin_sn
    generator_centroids = accretor.centroids
    print "N centroids", generator_centroids.shape[0]

    fig = Figure(figsize=(3.5, 3.5), frameon=False)
    canvas = FigureCanvas(fig)
    gs = gridspec.GridSpec(1, 1,
                           left=0.15, right=0.95, bottom=0.15, top=0.95,
                           wspace=None, hspace=None,
                           width_ratios=None, height_ratios=None)
    ax = fig.add_subplot(gs[0])
    ax.imshow(accretor.segimage, interpolation='None', aspect='equal')
    ax.scatter(generator_centroids[:, 0], generator_centroids[:, 1],
               c='k', marker='o', s=4)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_ylim(0, 255)
    ax.set_xlim(0, 255)
    gs.tight_layout(fig, pad=1.08, h_pad=None, w_pad=None, rect=None)
    canvas.print_figure("iso_sn_accretion.png", format="png")

    pix_sn = img / noise
    cvt = CVTessellation(pix_x.flatten(),
                         pix_y.flatten(),
                         pix_sn.flatten() ** 2.,
                         node_xy=generator_centroids)
    cvt_x = cvt.xNode
    cvt_y = cvt.yNode
    print "N nodes", len(cvt_x)
    cvt.set_pixel_grid((0, img.shape[1]),
                       (0, img.shape[0]))
    segimage = cvt.make_segmap()

    fig = Figure(figsize=(3.5, 3.5), frameon=False)
    canvas = FigureCanvas(fig)
    gs = gridspec.GridSpec(1, 1,
                           left=0.15, right=0.95, bottom=0.15, top=0.95,
                           wspace=None, hspace=None,
                           width_ratios=None, height_ratios=None)
    ax = fig.add_subplot(gs[0])
    ax.imshow(segimage, interpolation='None', aspect='equal')
    ax.scatter(cvt_x, cvt_y, c='k', marker='o', s=4)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_ylim(0, 255)
    ax.set_xlim(0, 255)
    gs.tight_layout(fig, pad=1.08, h_pad=None, w_pad=None, rect=None)
    canvas.print_figure("iso_sn_voronoi.png", format="png")


if __name__ == '__main__':
    main()
