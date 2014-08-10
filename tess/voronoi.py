#!/usr/bin/env python
# encoding: utf-8
"""
Representation of Voronoi Tessellations.
"""

import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import KDTree

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
        assert len(nodeValues) == self._sy.shape[0], "Not the same number of" \
            " node values as nodes!"

        # Pixel grid to compute Voronoi field on
        ygrid, xgrid = np.meshgrid(
            np.arange(self.ylim[0], self.ylim[1]),
            np.arange(self.xlim[0], self.xlim[1]))

        # Package xNode and yNode into Nx2 array
        # y is first index if FITS data is also structured this way
        yx = np.empty(self._xy.shape, dtype=self._xy.dtype)
        yx[:, 0] = self._xy[:, 1]
        yx[:, 1] = self._xy[:, 0]

        # Nearest neighbour interpolation is equivalent to Voronoi pixel
        # tessellation!
        return griddata(yx, nodeValues, (xgrid, ygrid), method='nearest')

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
    """
    def __init__(self, xy_points, dens_points, node_xy=None):
        x_node, y_node, vbin_num = self._tessellate(xy_points[:, 0],
                                                    xy_points[:, 1],
                                                    dens_points,
                                                    node_xy=node_xy)
        xy = np.column_stack((x_node, y_node))
        super(CVTessellation, self).__init__(xy)
        self._vbin_num = vbin_num

    @classmethod
    def from_image(cls, density, generators):
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
        """
        x, y = np.meshgrid(np.arange(density.shape[1], dtype=float),
                           np.arange(density.shape[0], dtype=float))
        instance = cls(x.flatten(),
                       y.flatten(),
                       density.flatten(),
                       node_xy=generators)
        instance.set_pixel_grid((0, density.shape[1]), (0, density.shape[0]))
        return instance

    def _tessellate(self, xPoints, yPoints, densPoints, node_xy=None):
        """Computes the centroidal voronoi tessellation itself."""
        self.densPoints = densPoints

        # Obtain pre-generator node coordinates
        if node_xy is None:
            node_xy = np.column_stack((xPoints.copy(), yPoints.copy()))

        xy = np.column_stack((xPoints, yPoints))
        node_xy, v_bin_numbers = lloyd(xy, densPoints, node_xy)
        return np.array(node_xy[:, 0]), np.array(node_xy[:, 1]), \
            np.array(v_bin_numbers)

    def _run_py_lloyds(self, xPoints, yPoints, densPoints, xNode, yNode):
        """Run Lloyd's algorithm in pure-python."""
        nPoints = len(xPoints)
        nNodes = len(xNode)

        # vBinNum holds the Voronoi bin numbers for each data point
        vBinNum = np.zeros(nPoints, dtype=np.uint32)

        iters = 1
        while 1:
            xNodeOld = xNode.copy()
            yNodeOld = yNode.copy()

            for j in xrange(nPoints):
                # Assign each point to a node. A point is assigned to the
                # node that it is closest to.
                # Note: this now means the voronoi bin numbers start from zero
                vBinNum[j] = np.argmin((xPoints[j] - xNode) ** 2.
                                       + (yPoints[j] - yNode) ** 2.)

            # Compute centroids of these Vorononi Bins. But now using a dens^2
            # weighting. The dens^2 weighting produces equal-mass Voronoi bins.
            # See Capellari and Copin (2003)
            for j in xrange(nNodes):
                indices = np.where(vBinNum == j)[0]
                if len(indices) != 0:
                    xBar, yBar = self._weighted_centroid(
                        xPoints[indices],
                        yPoints[indices], densPoints[indices] ** 2.)
                else:
                    # if the Voronoi bin is empty then give (0,0) as its
                    # centroid then we can catch these empty bins later
                    xBar = 0.0
                    yBar = 0.0
                xNode[j] = xBar
                yNode[j] = yBar

            delta = np.sum((xNode - xNodeOld) ** 2.
                           + (yNode - yNodeOld) ** 2.)
            iters = iters + 1

            print "CVT Iteration: %i, Delta %f" % (iters, delta)

            if delta == 0.:
                break
        print "CVT complete"
        return xNode, yNode, vBinNum

    def _weighted_centroid(self, x, y, density):
        """
        Compute the density-weighted centroid of one bin. See Equation 4 of
        Cappellari & Copin (2003).

        Parameters
        ----------
        x : ndarray
            Array of x-axis spatial coordinates
        y : ndarray
            Array of y-axis spatial coordinates
        density : ndarray
            Array of weighting values

        Returns
        -------
        centroid : tuple
            Weighted centroid, ``(x, y)``.
        """
        mass = np.sum(density)
        xBar = np.sum(x * density) / mass
        yBar = np.sum(y * density) / mass
        return (xBar, yBar)

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
