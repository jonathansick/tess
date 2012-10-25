#!/usr/bin/env python
# encoding: utf-8
"""
Representation of Voronoi Tessellations in ghostmap.

2012-10-25 - Created by Jonathan Sick
"""

import os
import numpy as np
import scipy.interpolate.griddata as griddata
import scipy.spatial.cKDTree as KDTree


class VoronoiTessellation(object):
    """A Voronoi Tessellation."""
    def __init__(self, x, y):
        super(VoronoiTessellation, self).__init__()
        self.xNode = x  #: Array of node x-coordinates
        self.yNode = y  #: Array of node y-coordinates
        self.segmap = None  #: 2D `ndarray` of `vBinNum` for each pixel
        self.cellAreas = None  #: 1D array of Voronoi cell areas

    def make_segmap(self, header=None, xlim=None, ylim=None):
        """Make a pixel segmentation map that paints the Voronoi bin number
        on Voronoi pixels.
        
        The result is stored as the `segmap` attribute and returned to the
        caller.
        
        :param header: pyfits header, used to define size of segmentation map.
        :param xlim, ylim: tuples of (min, max) pixel ranges, used if
                           `header` is `None`.
        :returns: The segmentation map array, `segmap`.
        """
        if header is not None:
            # Assume origin at 1, FITS standard
            xlim = (1, header['NAXIS2'] + 1)
            ylim = (1, header['NAXIS1'] + 1)
        else:
            assert xlim is not None, "Need a xlim range (min, max)"
            assert ylim is not None, "Need a ylim range (min, max)"
        xgrid = np.arange(xlim[0], xlim[1])
        ygrid = np.arange(ylim[0], ylim[1])
        # Package xNode and yNode into Nx2 array
        # y is first index if FITS data is also structured this way
        yxNode = np.hstack(self.yNode, self.xNode)
        # Nearest neighbour interpolation is equivalent to Voronoi pixel
        # tessellation!
        self.segmap = griddata(yxNode, np.arange(0, self.yNode.shape[0]),
                (xgrid, ygrid), method='nearest')

    def save_segmap(self, fitsPath, **kwargs):
        """Convenience wrapper to :meth:`make_segmap` that saves the
        segmentation map to a FITS file.

        :param fitsPath: full filename destination of FITS file
        :param kwargs: keyword arguments are passed to :meth:`make_segmap`.
        """
        import pyfits
        fitsDir = os.path.dirname(fitsPath)
        if not os.path.exists(fitsDir): os.makedirs(fitsDir)
        if self.segmap is None:
            self.make_segmap(**kwargs)
        if 'header' in kwargs:
            pyfits.writeto(fitsPath, self.segmap, kwargs['header'])
        else:
            pyfits.writeto(fitsPath, self.segmap)

    def compute_cell_areas(self):
        """Compute the areas of Voronoi cells; result in stored in the
        `self.cellAreas` attribute.

        .. note:: This method requires that the segmentation map is computed
           (see :meth:`make_segmap`), and is potentially expensive (I'm working
           on a faster implementation). Uses :func:`numpy.bincount` to count
           number of pixels in the segmentation map with a given Voronoi
           cell value. I'd prefer to calculate these from simple geometry,
           but no good python packages exist for defining Voronoi cell
           polygons.
        """
        assert self.segmap is not None, "Compute a segmentation map with first"
        pixelCounts = np.bincount(self.segmap.ravel())
        self.cellAreas = pixelCounts

    def get_nodes(self):
        """Returns the x and y positions of the Voronoi nodes."""
        return self.xNode, self.yNode

    def partition_points(self, x, y):
        """Partition an arbitrary set of points, defined by `x` and `y`
        coordinates, onto the Voronoi tessellation.
        
        This method uses :class:`scipy.spatial.cKDTree` to efficiently handle
        Voronoi assignment.

        :param x: array of point `x` coordinates
        :param y: array of point `y` coordinates
        :returns: ndarray of indices of Voronoi nodes
        """
        nodeData = np.hstack((self.nodeX, self.nodeY))
        pointData = np.hstack((x, y))
        tree = KDTree(nodeData)
        distances, indices = tree.query(pointData, k=1)
        return indices
    
    def plot_nodes(self, plotPath):
        """Plots the points in each bin as a different colour"""
        from matplotlib.backends.backend_pdf \
                import FigureCanvasPdf as FigureCanvas
        from matplotlib.figure import Figure
        
        fig = Figure(figsize=(6, 4))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.plot(self.xNode, self.yNode, 'ok')
        canvas.print_figure(plotPath)
