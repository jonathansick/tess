#!/usr/bin/env python
# encoding: utf-8
"""
Representation of Voronoi Tessellations.

The :class:`VoronoiTessellation` class  provides basic support for Voronoi
tessellations, partitioning points in Voronoi cells, and rendering Voronoi
fields. This class uses KD Trees to associate points and pixels to Voronoi
cells.

The :class:`CVTessellation` class is used to build a Voronoi tessellation by
finding the nodes the partition a data set into cells of equal mass. Once
built, a :class:`CVTessellation` provides all the same facilities as a
:class:`VoronoiTessellation`.

Note that the :class:`tess.delaunay.DelaunayTessellation` can also be used to
build a Voronoi tessellation and field rendering, though there are still bugs
in this approach.
"""

import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import KDTree

import logging
log = logging.getLogger(__name__)

from lloyd import lloyd


class VoronoiTessellation(object):
    """A Voronoi Tessellation, defined by a set of nodes on a 2D plane.

    Parameters
    ----------
    xy : ndarray, (n_nodes, 2)
        Array of node ``(x,y)`` coordinates.
    """
    def __init__(self, xy):
        super(VoronoiTessellation, self).__init__()
        self._xy = xy
        self._segmap = None  #: 2D `ndarray` of `vBinNum` for each pixel
        self._cell_areas = None  #: 1D array of Voronoi cell areas
        self.xlim = None  #: ``(min, max)`` coords of x pixel grid
        self.ylim = None  #: ``(min, max)`` coords of y pixel grid

    @property
    def nodes(self):
        """Voronoi tessellation nodes, a ``(n_points, 2)`` array."""
        return self._xy

    def set_pixel_grid(self, xlim, ylim):
        """Set a pixel grid bounding box for the tessellation. This is
        used when rendering Voronoi fields or computing cell areas.

        Setting the pixel grid is a prerequistie for using the :attr:`segmap`
        attribute and :meth:`compute_cell_areas` method.

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
        self._cell_areas = None

    @property
    def segmap(self):
        """Segmentation map of Voronoi bin numbers for each pixel."""
        if self._segmap is None:
            self._segmap = self.render_voronoi_field(
                np.arange(0, self._xy.shape[0]))
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
        assert len(nodeValues) == self._xy.shape[0], "Not the same number of" \
            " node values as nodes!"

        # Pixel grid to compute Voronoi field on
        ygrid, xgrid = np.meshgrid(
            np.arange(self.ylim[0], self.ylim[1]),
            np.arange(self.xlim[0], self.xlim[1]))

        # Nearest neighbour interpolation is equivalent to Voronoi pixel
        # tessellation!
        return griddata(self._xy, nodeValues, (xgrid, ygrid),
                        method='nearest').T

    def compute_cell_areas(self, flagmap=None):
        """Compute the areas of Voronoi cells; result is stored in the
        `self._cell_areas` attribute.

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
        self._cell_areas = pixelCounts
        return self._cell_areas

    def partition_points(self, xy):
        """Partition an arbitrary set of points, defined by `x` and `y`
        coordinates, onto the Voronoi tessellation.

        This method uses :class:`scipy.spatial.cKDTree` to efficiently handle
        Voronoi assignment.

        Parameters
        ----------
        xy : ndarray, ``(n_points, 2)``
            Array of point ``(x,y)`` coordinates

        Returns
        -------
        indices : ndarray
            Array of indices of Voronoi nodes
        """
        tree = KDTree(self._xy)
        distances, indices = tree.query(xy, k=1)
        return indices

    def sum_cell_point_mass(self, xy, mass=None):
        """Given a set of points with masses, computes the mass within
        each Voronoi cell.

        Parameters
        ----------
        xy : ndarray, ``(n_points, 2)``
            Array of point ``(x,y)`` coordinates
        mass : ndarray
            Mass of each point. If `None`, then each point is assumed to have
            unit mass.

        Returns
        -------
        mass : ndarray
            Sum of masses of points within each Voronoi cell.
        """
        if mass is None:
            mass = np.ones(xy.shape[0])
        cellIndices = self.partition_points(xy)
        cellMass = np.bincount(cellIndices, weights=mass)
        return cellMass

    def cell_point_density(self, xy, mass=None, flagmap=None):
        """Compute density of points in each Voronoi cell.

        .. note:: This method calls :meth:`compute_cell_areas` if the cell
           areas have not been compute yet. The `flagmap` parameter can be
           passed to that method. If :attr:`cellAreas` is not `None`,
           then new cell areas will *not* be computed.

        Parameters
        ----------
        xy : ndarray, ``(n_points, 2)``
            Array of point ``(x,y)`` coordinates
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
        if self._cell_areas is None:
            self.compute_cell_areas(flagmap=flagmap)
        return self.sum_cell_point_mass(xy, mass=mass) / self._cell_areas


class CVTessellation(VoronoiTessellation):
    """A centroidal Voronoi tessellation (CVT) Uses Lloyd's algorithm to assign
    data points to Voronoi bins so that each bin has an equal mass.

    The :mod:`tess.pixel_accretion` and :mod:`tess.point_accretion` modules
    are useful for building node coordinates to seed the CVT.

    Inherits from :class:`tess.voronoi.VoronoiTessellation`.

    Parameters
    ----------
    xy_points : ndarray, ``(n_points, 2)``
        Array of cartesian ``(x,y)`` coordinates of each data point.
    dens_points : ndarray
        Density *or weight* of each point. For an equal-S/N generator, this
        should be set to :math:`(S/N)^2`. For an equal number generator this
        can be simple an array of ones.
    node_xy : ndarray ``(n_nodes, 2)``
        A ``(n_points, 2)`` array of coordinates of pre-computed generators
        for the tessellation. You can use
        :class:`tess.point_accretion.PointAccretion` and subclasses to build
        an array of generators accordinate to target mass or S/N.
    max_iters : int
        Maximum number of iterations of Lloyd's algorithm.
    """
    def __init__(self, xy_points, dens_points, node_xy=None, max_iters=300):
        xy, vbin_num = self._tessellate(xy_points,  # CHANGED
                                        dens_points,
                                        node_xy=node_xy,
                                        max_iters=max_iters)
        super(CVTessellation, self).__init__(xy)
        self._vbin_num = vbin_num

    @classmethod
    def from_image(cls, density, generators, max_iters=300):
        """Convenience constructor for centroidal Voronoi tessellations
        of pixel data sets.

        The (x, y) point coordinates are automatically set to be the 0-based
        pixel indices. :meth:`CVTessellation.set_pixel_grid` is automatically
        called to set the pixel grid to the match the image dimensions.

        Parameters
        ----------
        density : ndarray
            A 2D image with the density. The CVT partitions the density
            map so each cell has approximately equal mass.
        generators : ndarray
            A ``(n_nodes, 2)`` array of point coordinates with initial
            starting points for each Voronoi cell.
            Note that coordinates are (x, y), which is the reverse of (y, x)
            image indices.
        max_iters : int
            Maximum number of iterations of Lloyd's algorithm.
        """
        x, y = np.meshgrid(np.arange(density.shape[1], dtype=float),
                           np.arange(density.shape[0], dtype=float))
        xy = np.column_stack((x.flatten(), y.flatten()))
        dens = density.flatten()
        good = np.where(np.isfinite(dens))[0]
        instance = cls(xy[good, :],
                       dens[good],
                       node_xy=generators,
                       max_iters=max_iters)
        instance.set_pixel_grid((0, density.shape[1]), (0, density.shape[0]))
        return instance

    def _tessellate(self, xy, densPoints, node_xy=None, max_iters=300):
        """Computes the centroidal voronoi tessellation itself."""
        self.densPoints = densPoints

        # Obtain pre-generator node coordinates
        if node_xy is None:
            node_xy = xy.copy()

        node_xy, v_bin_numbers, converged = lloyd(xy, densPoints, node_xy,
                                                  max_iters)
        if not converged:
            log.warning("CVT did not converge")
        return np.asarray(node_xy), np.array(v_bin_numbers)

    @property
    def membership(self):
        """Array of indices into Voronoi bins for each point."""
        return self._vbin_num

    @property
    def node_weights(self):
        """Weight of each Voronoi bin (sum of enclosed point masses)."""
        nNodes = self._xy.shape[0]
        nodeWeights = np.zeros(nNodes, dtype=np.float)
        for i in xrange(nNodes):
            ind = np.where(self._vbin_num == i)[0]
            if len(ind) > 0:
                nodeWeights[i] = np.sum(self.densPoints[ind])
        return nodeWeights
