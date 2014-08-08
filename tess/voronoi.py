#!/usr/bin/env python
# encoding: utf-8
"""
Representation of Voronoi Tessellations.
"""

import os
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import KDTree


class VoronoiTessellation(object):
    """A Voronoi Tessellation, defined by a set of nodes on a 2D plane.

    Parameters
    ----------
    x : ndarray, (n_nodes, 1)
        Array of node x-coordinates.
    y : ndarray, (n_nodes, 1)
        Array of node y-coordinates.
    """
    def __init__(self, x, y):
        super(VoronoiTessellation, self).__init__()
        self.xNode = x  #: Array of node x-coordinates
        self.yNode = y  #: Array of node y-coordinates
        self._segmap = None  #: 2D `ndarray` of `vBinNum` for each pixel
        self.cellAreas = None  #: 1D array of Voronoi cell areas
        self.xlim = None  #: Length-2 array of min, max coords of x pixel grid
        self.ylim = None  #: Length-2 array of min, max coords of y pixel grid
        self.header = None  #: a PyFITS header representing pixel grid

    def set_pixel_grid(self, xlim, ylim):
        """Set a pixel grid bounding box for the tessellation. This is
        used when rendering Voronoi fields or computing cell areas.

        Setting the pixel grid is a prerequistie for running the methods:

        - :meth:`make_segmap` and :meth:`save_segmap`
        - :meth:`compute_cell_areas`

        Parameters
        ----------
        xlim : tuple
            Tuple of (min, max) pixel range along x-axis.
        ylim : tuple
            Tuple of of (min, max) pixel range along y-axis.
        """
        assert len(xlim) == 2, "xlim must be (min, max) sequence"
        assert len(ylim) == 2, "ylim must be (min, max) sequence"
        self.xlim = xlim
        self.ylim = ylim

        # Reset dependencies
        self._segmap = None
        self.cellAreas = None

    def set_fits_grid(self, header):
        """Convenience wrapper to :meth:`set_pixel_grid` if a FITS header is
        available. As a bonus, the FITS header will be used when saving
        any rendered fields to FITS.

        .. note:: The header is available as the :attr:`header` attribute.

        .. todo:: Should be removed

        Parameters
        ----------
        header : :class:`astropy.io.fits.Header`
            An astropy FITS image header; defines area of rendered Voronoi
            images (e.g. segmentaion maps or fields).
        """
        # Assume origin at 1, FITS standard
        xlim = (1, header['NAXIS2'] + 1)
        ylim = (1, header['NAXIS1'] + 1)
        self.set_pixel_grid(xlim, ylim)
        self.header = header

    @property
    def segmap(self):
        """Segmentation map of Voronoi bin numbers for each pixel."""
        if self._segmap is None:
            self._segmap = self.render_voronoi_field(
                np.arange(0, self.yNode.shape[0]))
        return self._segmap

    def render_voronoi_field(self, nodeValues):
        """Renders the Voronoi field onto the pixel context with the given
        `nodeValues` for each Voronoi cell.

        .. note:: Must set the pixel grid context with
           either :meth:`set_pixel_grid` or :meth:`set_fits_grid` first!

        Parameters
        ----------
        nodeValues : ndarray
            1D array of values for Voronoi nodes (must be same length as
            :attr:`xNode` and :attr:`xNode`.

        Returns
        -------
        field : ndarray
            2D array (image) of Voronoi field.
        """
        assert self.xlim is not None, "Need to run `set_pixel_grid()` first"
        assert self.ylim is not None, "Need to run `set_pixel_grid()` first"
        assert len(nodeValues) == len(self.xNode), "Not the same number of" \
            " node values as nodes!"

        # Pixel grid to compute Voronoi field on
        ygrid, xgrid = np.meshgrid(
            np.arange(self.ylim[0], self.ylim[1]),
            np.arange(self.xlim[0], self.xlim[1]))

        # Package xNode and yNode into Nx2 array
        # y is first index if FITS data is also structured this way
        yxNode = np.vstack((self.yNode, self.xNode)).T

        # Nearest neighbour interpolation is equivalent to Voronoi pixel
        # tessellation!
        return griddata(yxNode, nodeValues, (xgrid, ygrid), method='nearest')

    def save_segmap(self, fitsPath):
        """Convenience wrapper to :meth:`make_segmap` that saves the
        segmentation map to a FITS file.

        TODO: This should be removed.

        Parameters
        ----------
        fitsPath : str
            Full filename destination of FITS file.
        """
        import astropy.io.fits
        fitsDir = os.path.dirname(fitsPath)
        if fitsDir is not '' and fitsDir is not os.path.exists(fitsDir):
            os.makedirs(fitsDir)
        if self._segmap is None:
            self.make_segmap()
        if self.header is not None:
            astropy.io.fits.writeto(
                fitsPath, self._segmap, self.header,
                clobber=True)
        else:
            astropy.io.fits.writeto(fitsPath, self._segmap, clobber=True)

    def compute_cell_areas(self, flagmap=None):
        """Compute the areas of Voronoi cells; result is stored in the
        `self.cellAreas` attribute.

        .. note:: This method requires that the segmentation map is computed
           (see :meth:`make_segmap`), and is potentially expensive (I'm working
           on a faster implementation). Uses :func:`numpy.bincount` to count
           number of pixels in the segmentation map with a given Voronoi
           cell value. I'd prefer to calculate these from simple geometry,
           but no good python packages exist for defining Voronoi cell
           polygons.

        Parameters
        ----------
        flagmap : ndarray
            Any pixels in the flagmap with values greater than zero will be
            omitted from the area count. Thus the cell areas will report
            *useable* pixel areas, rather than purely geometric areas. This is
            useful to avoid bias in density maps due to 'bad' pixels.

        Returns
        -------
        cellAreas : ndarray
            Array of cell areas (square pixels). This array is also
            stored as :attr:`cellAreas`.
        """
        assert self._segmap is not None, "Compute a segmentation map first"

        if flagmap is not None:
            # If a flagmap is available, flagged pixels are set to NaN
            _segmap = self._segmap.copy()
            _segmap[flagmap > 0] = np.nan
        else:
            _segmap = self._segmap
        pixelCounts = np.bincount(_segmap.ravel())
        self.cellAreas = pixelCounts
        return self.cellAreas

    def get_nodes(self):
        """Returns the x and y positions of the Voronoi nodes.

        Returns
        -------
        xNode : ndarray
            X-coordinates of Voronoi nodes.
        yNode : ndarray
            Y-coordinates of Voronoi nodes
        """
        return self.xNode, self.yNode

    def partition_points(self, x, y):
        """Partition an arbitrary set of points, defined by `x` and `y`
        coordinates, onto the Voronoi tessellation.

        This method uses :class:`scipy.spatial.cKDTree` to efficiently handle
        Voronoi assignment.

        Parameters
        ----------
        x : ndarray
            Array of point `x` coordinates
        y : ndarray
            Array of point `y` coordinates

        Returns
        -------
        indices : ndarray
            Array of indices of Voronoi nodes
        """
        nodeData = np.vstack((self.xNode, self.yNode)).T
        pointData = np.vstack((x, y)).T
        tree = KDTree(nodeData)
        distances, indices = tree.query(pointData, k=1)
        return indices

    def sum_cell_point_mass(self, x, y, mass=None):
        """Given a set of points with masses, computes the mass within
        each Voronoi cell.

        Parameters
        ----------
        x : ndarray
            X-coordinates of points to assign to Voronoi cells.
        y : ndarray
            Y-coordinates of points to assign to Voronoi cells.
        mass : ndarray
            Mass of each point. If `None`, then each point is assumed to have
            unit mass.

        Returns
        -------
        mass : ndarray
            Sum of masses of points within each Voronoi cell.
        """
        if mass is None:
            mass = np.ones(len(x))
        cellIndices = self.partition_points(x, y)
        cellMass = np.bincount(cellIndices, weights=mass)
        return cellMass

    def cell_point_density(self, x, y, mass=None, flagmap=None):
        """Compute density of points in each Voronoi cell.

        .. note:: This method calls :meth:`compute_cell_areas` if the cell
           areas have not been compute yet. The `flagmap` parameter can be
           passed to that method. If :attr:`cellAreas` is not `None`,
           then new cell areas will *not* be computed.

        Parameters
        ----------
        x : ndarray
            1D array of point x-coordinates
        y : ndarray
            1D array of point y-coordinates
        mass : ndarray
            Optional 1D array of point masses (or *weights*). If `None`, then
            each point is assumed to have unit mass.
        flagmap : ndarray
            Optional flagmap to be passed to :meth:`compute_cell_areas`.

        Returns
        -------
        density : ndarray
            Density of each Voronoi cell, in units of mass / square pixel.
        """
        if self.cellAreas is None:
            self.compute_cell_areas(flagmap=flagmap)
        return self.sum_cell_point_mass(x, y, mass=mass) / self.cellAreas

    def plot_nodes(self, plotPath):
        """Plots the points in each bin as a different colour

        Parameters
        ----------
        plotPath : str
            Path where the plot will be saved.
        """
        from matplotlib.backends.backend_pdf \
            import FigureCanvasPdf as FigureCanvas
        from matplotlib.figure import Figure

        fig = Figure(figsize=(6, 4))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.plot(self.xNode, self.yNode, 'ok')
        if self.xlim is not None:
            ax.set_xlim(self.xlim)
        if self.ylim is not None:
            ax.set_ylim(self.ylim)
        canvas.print_figure(plotPath)
