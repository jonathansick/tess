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
    # Prepare mock data set with an image of a 2D Gaussian and constant noise
    shape = (256, 256)
    pix_x, pix_y = np.meshgrid(np.arange(shape[1], dtype=float),
                               np.arange(shape[0], dtype=float))
    img = 10. * np.exp(-((pix_x - 128.) ** 2. / (2. * 50. ** 2.) +
                         (pix_y - 128.) ** 2. / (2. * 50. ** 2.)))
    noise = np.ones(shape, dtype=float)
    pix_sn = img / noise

    # Accrete pixels into equal S/N bins to make a set of Voronoi generators
    accretor = EqualSNAccretor(img, noise, 50., start=(0, 0))
    accretor.cleanup()
    generator_centroids = accretor.centroids

    plot_segmap(accretor.segimage, generator_centroids, "iso_sn_accretion")

    # Make a centroidal Voronoi tessellation so that each cell has
    # approximately equal S/N
    cvt = CVTessellation.from_image(pix_sn ** 2., generator_centroids)
    cvt_xy = cvt.nodes
    segimage = cvt.segmap

    plot_segmap(segimage, cvt_xy, "iso_sn_voronoi")


def plot_segmap(segmap, nodes, plot_path):
    fig = Figure(figsize=(3.5, 3.5), frameon=False)
    canvas = FigureCanvas(fig)
    gs = gridspec.GridSpec(1, 1,
                           left=0.15, right=0.95, bottom=0.15, top=0.95,
                           wspace=None, hspace=None,
                           width_ratios=None, height_ratios=None)
    ax = fig.add_subplot(gs[0])
    ax.imshow(segmap, interpolation='None', aspect='equal')
    ax.scatter(nodes[:, 0], nodes[:, 1], c='k', marker='o', s=4)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_ylim(0, 255)
    ax.set_xlim(0, 255)
    gs.tight_layout(fig, pad=1.08, h_pad=None, w_pad=None, rect=None)
    canvas.print_figure("iso_sn_voronoi.png", format="png")


if __name__ == '__main__':
    main()
