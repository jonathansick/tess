#!/usr/bjn/env python
# encoding: utf-8
"""
Density estimation tools.
"""

import numpy as np


class DelaunayDensityEstimator(object):
    """Uses the DTFE to set the density of each Delaunay node from a
    DelaunayTessellation and the point data. This can then be used as the
    values for the first-order Delaunay density field reconstruction.

    Parameters
    ----------
    delaunay : :class:`tess.delaunay.DelaunayTessellation` instance
        A :class:`tess.delaunay.DelaunayTessellation` object, which has the
        Delaunay triangulation
    """
    def __init__(self, delaunay):
        super(DelaunayDensityEstimator, self).__init__()
        self.delaunay = delaunay
        self.node_density = None
        # hullCorrection: if True, then correct the incomplete area around
        # Delaunay vertices on point distribution's hull
        # NOTE: this seems to be broken
        self.hullCorrection = False

    def estimate_density(self, xRange, yRange, nodeMasses, pixelMask=None):
        """Estimate the density of each node in the delaunay tessellation using
        the DTFE of Schaap (PhD Thesis, Groningen).

        .. todo:: Implement masked area checking

        .. todo:: Implement hull correction

        Parameters
        ----------
        xRange : tuple
            A tuple of (x_min, x_max); TODO make these optional, just to scale
            the pixel mask array
        yRange : tuple
            A tuple of (y_min, y_max)
        nodeMasses : ndarray, `(n_nodes,)`
            An array of the mass of each node, in the same order used by the
            `delaunay` object
        pixelMask : ndarray, `(ny, nx)`
            Numpy array of same size as the extent of `xRange` and `yRange`.
            A pixel value of 1 is masked, values of 0 are admitted. This allows
            masked areas to be excluded from the density computation.
        """
        xNode = self.delaunay.nodes[:, 0]
        yNode = self.delaunay.nodes[:, 1]
        nNodes = len(xNode)
        if nNodes != len(nodeMasses):
            print "Warning: nodeMasses has wrong length (%i, should be %i)" % \
                (len(xNode), len(nodeMasses))

        memTable = self.delaunay.membership_table
        nTri = self.delaunay.n_triangles
        areas = self.delaunay.triangle_areas
        extremeNodes = self.delaunay.hull_nodes

        # TODO this is where I would compute the masked areas...
        # use the Walking Triangle algorithm
        maskedAreas = np.zeros(nTri)
        if pixelMask is not None:
            pass

        # Compute the area of triangles contiguous about each node
        contigAreas = np.zeros([nNodes])
        for i in xrange(nNodes):
            # list of triangle indices associated with node i
            contigTris = memTable[i]
            contigAreas[i] = areas[contigTris].sum() \
                - maskedAreas[contigTris].sum()

        # Correct the area of contiguous Voronoi regions on the outside of the
        # convex hull.
        # TODO check this
        if self.hullCorrection:
            nExtremeNodes = len(extremeNodes)
            for i, node in enumerate(extremeNodes):
                # find the neighbouring extreme points, the one to the left
                # and right of the point being studied
                if i > 0:
                    rightNode = extremeNodes[i-1]
                else:
                    # take the wrap-around
                    rightNode = extremeNodes[nExtremeNodes-1]
                if i < nExtremeNodes-1:
                    leftNode = extremeNodes[i+1]
                else:
                    leftNode = extremeNodes[0]
                # find the angle that they subtend, using The Law of Cosines
                a = np.sqrt(
                    (xNode[leftNode] - xNode[node]) ** 2 +
                    (yNode[leftNode] - yNode[node]) ** 2)
                b = np.sqrt(
                    (xNode[rightNode] - xNode[node]) ** 2 +
                    (yNode[rightNode] - yNode[node]) ** 2)
                c = np.sqrt(
                    (xNode[rightNode] - xNode[leftNode]) ** 2 +
                    (yNode[rightNode] - yNode[node]) ** 2)
                subAngle = np.acos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b))
                # The sub angle should in theory be less than 180 degrees.
                # This code ensures that we have an angle that covers more
                # than 180 degrees.
                extraAngle = 2 * np.pi - subAngle
                if extraAngle < np.pi:
                    print "Angle error in edge effect correction"
                correctionFactor = extraAngle / subAngle
                # update the contiguous area:
                contigAreas[node] = (1. + correctionFactor) * contigAreas[node]

        # Finally compute the density at the site of each node by using
        # eqn 3.36 of Schaap 2007 (pg 69)
        self.node_density = 3. * nodeMasses / contigAreas

        # Compute the total tessellated area
        self.totalArea = contigAreas.sum() / 3.

        return self.node_density


def rectangular_density_field(x, y, mass, x_range, y_range, x_binsize,
                              y_binsize):
    """Performs rectangular binning on a point distribution to yield a density
    map.

    Returns a field with pixels in units of sum(mass_points)/area. Each pixel
    in the field is a bin. If you want a bin to occupy more than one pixel,
    just resize the np array.

    Parameters
    ----------
    x : ndarray
        Array of x point coordinates
    y : ndarray
        Array of y point coordinates
    mass : ndarray
        Array of point masses
    x_range : tuple
        Tuple of (xmin, xmax); determines range of reconstructed field
    y_range : tuple
        Tuple of (ymin, ymax)
    x_binsize : scalar
        Length of bins along x-axis
    y_binsize : scalar
        Length of bins along y-axis

    Returns
    -------
    field : ndarray
        2D array (image) of the rectangular binned density field.
    """
    xGrid = np.arange(min(x_range), max(x_range) - x_binsize, x_binsize)
    yGrid = np.arange(min(y_range), max(y_range) - y_binsize, y_binsize)
    field = np.zeros([len(yGrid), len(xGrid)])
    binArea = x_binsize * y_binsize

    # Trim the dataset to ensure it only covers the range of the field
    # reconstruction
    good = np.where((x > min(x_range))
                    & (x < max(x_range))
                    & (y > min(y_range))
                    & (y < max(y_range)))[0]
    x = x[good]
    y = y[good]
    mass = mass[good]

    # Then sort the point into increasing x-coordinate
    xsort = np.argsort(x)
    x = x[xsort]
    y = y[xsort]
    mass = mass[xsort]

    for xi, xg in enumerate(xGrid):
        col = np.where((x > xg) & (x < (xg + x_binsize)))[0]
        if len(col) == 0:
            continue  # no points in whole column
        ycol = y[col]
        mcol = mass[col]
        for yi, yg in enumerate(yGrid):
            bin = np.where((ycol > yg) & (ycol < (yg + y_binsize)))[0]
            if len(bin) == 0:
                continue  # no points in bin
            totalMass = np.sum(mcol[bin])
            field[yi, xi] = totalMass

    # Make it a density plot by dividing each pixel by the binArea
    field = field / binArea

    return field
