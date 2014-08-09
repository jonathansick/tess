#!/usr/bin/env python
# encoding: utf-8
"""
Demonstration of point density estimation.
"""

import numpy as np

# import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.gridspec as gridspec
from matplotlib.collections import PolyCollection

from tess.point_accretion import EqualMassAccretor
from tess.cvtessellation import CVTessellation
from tess.delaunay import DelaunayTessellation
from tess.density import DelaunayDensityEstimator


def main():
    img_shape = (1024, 1024)
    mean = (512., 512.)
    cov = 50000. * np.array(((1., 0.5), (0.5, 1.)))
    point_xy = np.random.multivariate_normal(mean, cov, 10000)
    s = np.where((point_xy[:, 0] >= 0.) &
                 (point_xy[:, 1] >= 0.) &
                 (point_xy[:, 0] < img_shape[0] - 1.) &
                 (point_xy[:, 1] < img_shape[1] - 1.))[0]
    point_xy = point_xy[s, :]
    point_mass = np.ones(point_xy.shape[0])

    plot_points(point_xy, img_shape, "density_demo_points")

    # Bin points with a CVT
    accretor = EqualMassAccretor(point_xy, point_mass, 100.)
    accretor.accrete()
    generator_xy = accretor.nodes()
    cvt = CVTessellation(point_xy[:, 0], point_xy[:, 1], point_mass,
                         node_xy=generator_xy)
    node_xy = cvt.nodes

    # FIXME issue with this rendering
    plot_voronoi_tessellation(cvt, img_shape, "density_demo_voronoi_cells")

    # Density estimate
    delaunay = DelaunayTessellation(node_xy[:, 0], node_xy[:, 1])
    dtfe = DelaunayDensityEstimator(delaunay)
    dens = dtfe.estimate_density((0., img_shape[1]),
                                 (0., img_shape[0]),
                                 cvt.node_weights)

    del_dens_map = delaunay.render_delaunay_field(dens,
                                                  (0., img_shape[1]),
                                                  (0., img_shape[0]),
                                                  1., 1.,)

    # FIXME this is broken too
    vor_dens_map = delaunay.render_voronoi_field(dens,
                                                 (0., img_shape[1]),
                                                 (0., img_shape[0]),
                                                 1., 1.,)

    plot_density_map(del_dens_map, "density_demo_delaunay_field")
    plot_density_map(vor_dens_map, "density_demo_voronoi_field")


def plot_points(xy, img_shape, plot_path):
    fig = Figure(figsize=(3.5, 3.5), frameon=False)
    canvas = FigureCanvas(fig)
    gs = gridspec.GridSpec(1, 1,
                           left=0.15, right=0.95, bottom=0.15, top=0.95,
                           wspace=None, hspace=None,
                           width_ratios=None, height_ratios=None)
    ax = fig.add_subplot(gs[0])
    ax.scatter(xy[:, 1], xy[:, 0], c='k', s=2)
    ax.set_xlim(0., img_shape[1])
    ax.set_ylim(0., img_shape[0])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    gs.tight_layout(fig, pad=1.08, h_pad=None, w_pad=None, rect=None)
    canvas.print_figure("{0}.png".format(plot_path), format="png")


def plot_voronoi_tessellation(cvt, img_shape, plot_path):
    xy = cvt.nodes
    delaunay = DelaunayTessellation(xy[:, 0], xy[:, 1])

    fig = Figure(figsize=(3.5, 3.5), frameon=False)
    canvas = FigureCanvas(fig)
    gs = gridspec.GridSpec(1, 1,
                           left=0.15, right=0.95, bottom=0.15, top=0.95,
                           wspace=None, hspace=None,
                           width_ratios=None, height_ratios=None)
    ax = fig.add_subplot(gs[0])

    polys = PolyCollection(delaunay.voronoi_vertices,
                           facecolors='None', edgecolors='k')
    ax.add_collection(polys)

    ax.scatter(xy[:, 1], xy[:, 0], c='k', s=2)

    ax.set_xlim(0., img_shape[1])
    ax.set_ylim(0., img_shape[0])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    gs.tight_layout(fig, pad=1.08, h_pad=None, w_pad=None, rect=None)
    canvas.print_figure("{0}.png".format(plot_path), format="png")


def plot_density_map(dens_map, plot_path):
    img_shape = dens_map.shape

    fig = Figure(figsize=(3.5, 3.5), frameon=False)
    canvas = FigureCanvas(fig)
    gs = gridspec.GridSpec(1, 1,
                           left=0.15, right=0.95, bottom=0.15, top=0.95,
                           wspace=None, hspace=None,
                           width_ratios=None, height_ratios=None)
    ax = fig.add_subplot(gs[0])

    ax.imshow(dens_map)

    ax.set_xlim(0., img_shape[1])
    ax.set_ylim(0., img_shape[0])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    gs.tight_layout(fig, pad=1.08, h_pad=None, w_pad=None, rect=None)
    canvas.print_figure("{0}.png".format(plot_path), format="png")


if __name__ == '__main__':
    main()
