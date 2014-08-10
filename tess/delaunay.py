#!/usr/bin/env python
# encoding: utf-8
"""
Classes for represent a spatial dataset as a Delaunay Tessellation and
performing a density analysis.
"""

import math
import numpy as np
from matplotlib.delaunay.triangulate import Triangulation
from PIL import Image, ImageDraw


class DelaunayTessellation(object):
    """Creates a Delaunay triangulation of the given nodes.

    Parameters
    ----------
    xNode : ndarray
        A ``(n_points, 1)`` array of x-coordinates of each Voronoi node.
    yNode : ndarray
        A ``(n_points, 1)`` array of y-coordinates of each Voronoi node.
    """
    def __init__(self, xNode, yNode):
        super(DelaunayTessellation, self).__init__()
        self.xNode = xNode
        self.yNode = yNode
        self._triangulation = Triangulation(self.xNode, self.yNode)
        self._mem_table = None

    @property
    def triangulation(self):
        """The :class:`matplotlib.delaunay.triangulate.Triangulation` instance.
        """
        return self._triangulation

    @property
    def nodes(self):
        """Coordinates of Voronoi nodes, a ``(n_nodes, 2)`` numpy array."""
        return np.column_stack((self.xNode, self.yNode))

    @property
    def hull_nodes(self):
        """Indices of nodes on the convex hull of the node distribution."""
        return self._triangulation.hull

    @property
    def triangles(self):
        """Get node indices of each triangle, a ``(n_tri, 3)`` array."""
        return self._triangulation.triangle_nodes

    @property
    def adjacency_matrix(self):
        """Gets matrix describing the neighbouring triangles of each triangle.

        .. note:: As per Robert Kern's Delaunay package: The value can also be
           ``-1`` meaning that that edge is on the convex hull of the points
           and there is no neighbor on that edge. The values are ordered such
           that ``triangle_neighbors[tri, i]`` corresponds with the edge
           *opposite* ``triangle_nodes[tri, i]``. As such, these neighbors are
           also in counter-clockwise order.
        """
        return self._triangulation.triangle_neighbors

    @property
    def circumcenters(self):
        """Array, ``(n_tri, 2) of the circumcenter of each Delaunay triangle.
        """
        return self._triangulation.circumcenters

    @property
    def membership_table(self):
        """Indices of all triangles associated (length ``n_nodes`` long; each
        item is another list giving triangle indices).
        """
        if self._mem_table is None:
            self._make_membership_table()
        return self._mem_table

    @property
    def n_triangles(self):
        """Number of triangles in the tessellation."""
        return self._triangulation.triangle_nodes.shape[0]

    @property
    def triangle_areas(self):
        """Array of geometric areas of each triangle."""
        # Make abbreviated pointers
        tri = self.triangles
        x = self.xNode
        y = self.yNode
        # Compute side lengths
        a = np.sqrt(
            (x[tri[:, 1]] - x[tri[:, 0]]) ** 2. +
            (y[tri[:, 1]] - y[tri[:, 0]]) ** 2.)  # between verts 1 and 0
        b = np.sqrt(
            (x[tri[:, 2]] - x[tri[:, 1]]) ** 2. +
            (y[tri[:, 2]] - y[tri[:, 1]]) ** 2.)  # between verts 2 and 1
        c = np.sqrt(
            (x[tri[:, 2]] - x[tri[:, 0]]) ** 2. +
            (y[tri[:, 2]] - y[tri[:, 0]]) ** 2.)  # between verts 2 and 0
        s = (a + b + c) / 2.  # semi-perimeter
        areas = np.sqrt(s * (s - a) * (s - b) * (s - c))  # Heron's formula
        return areas

    @property
    def voronoi_vertices(self):
        """Lists of polygon vertices for the Voronoi cells surrounding each
        node.

        These vertices are ordered counter clockwise.
        """
        # Function to assess the radial location of a point
        # Based on http://stackoverflow.com/questions/
        #          1709283/how-can-i-sort-a-coordinate-list-for-a-rectangle
        #          -counterclockwise
        ccwDist = lambda xy: \
            (math.atan2(xy[1] - y0, xy[0] - x0) + 2. * math.pi) % 2. * math.pi

        xNode = self.nodes[:, 0]
        yNode = self.nodes[:, 1]
        memTable = self.membership_table
        nExtremeNodes = len(self.hull_nodes)
        cells = []
        for i, members in enumerate(memTable):
            # voronoi vertices are the circumcenters of triangles the node
            # belongs to
            voronoiCell = [self.circumcenters[j] for j in members]
            # if the node is on the convext hull... then add as points the
            # bisections of of the neighbouring hull edges and the node itself.
            if i in self.hull_nodes:
                print "%i is on hull; there are %i nodes on hull" \
                    % (i, nExtremeNodes)
                # find the neighbouring extreme points, the one to the left
                # and right of the point being studied
                k = self.hull_nodes.index(i)  # index into extreme nodes list
                if k > 0:
                    rightNode = self.hull_nodes[k - 1]
                else:
                    # take the wrap-around
                    rightNode = self.hull_nodes[nExtremeNodes - 1]
                if k < nExtremeNodes - 1:
                    leftNode = self.hull_nodes[k + 1]
                else:
                    leftNode = self.hull_nodes[0]
                # Now left and right nodes are indices into the nodes list
                # of the node's hull neighbours
                print "\tneighbours: %i and %i" % (leftNode, rightNode)
                voronoiCell.append(((xNode[rightNode] + xNode[i]) / 2.,
                                    (yNode[rightNode] + yNode[i]) / 2.))
                voronoiCell.append(((xNode[leftNode] + xNode[i]) / 2.,
                                    (yNode[leftNode] + yNode[i]) / 2.))
                voronoiCell.append((xNode[i], yNode[i]))
            # Find the geometric centre of the cell
            nP = len(voronoiCell)  # number of vertices of the voronoi polygon
            x0 = sum(p[0] for p in voronoiCell) / nP
            y0 = sum(p[1] for p in voronoiCell) / nP
            voronoiCell.sort(key=ccwDist)  # ensure the vertices are sorted CCW
            cells.append(voronoiCell)
        return cells

    def _make_membership_table(self):
        """Computes the membership table of the triangles that each node is
        associated with."""
        n_nodes = self.nodes.shape[0]
        self._mem_table = [None] * n_nodes
        nTri = self.n_triangles
        vertices = range(3)
        for i in xrange(nTri):
            for j in vertices:
                theNode = self.triangles[i, j]
                if self._mem_table[theNode] is None:
                    self._mem_table[theNode] = [i]
                else:
                    # check to see if the triangle is noted for theNode yet
                    triExists = False
                    for k in range(len(self._mem_table[theNode])):
                        if i == self._mem_table[theNode][k]:
                            triExists = True
                    if not triExists:
                        self._mem_table[theNode].append(i)

    def render_voronoi_field(self, node_values, x_range, y_range,
                             x_step, y_step):
        """Renders a zeroth-order field (a Voronoi tiling).

        Parameters
        ----------
        node_values : ndarray ``(n_nodes, 1)``
             Array of the field values at each node
        x_range : tuple
            Tuple of ``(x_min, x_max)``
        y_range : tuple
            Tuple of ``(y_min, y_max)``
        x_step : float
            Scalar, size of pixels along x-axis
        y_step : float
            Scalar, size of pixels along y-axis

        Returns
        -------
        field : ndarray, ``(ny, nx)``
            2D array (image) zeroth-order Voronoi field.
        """
        xScalarRange = x_range[1] - x_range[0]
        yScalarRange = y_range[1] - y_range[0]

        nX = int(xScalarRange / x_step)
        nY = int(yScalarRange / y_step)

        # Transform the cell vertices in physical units to the pixel units of
        # the rendering space
        pixVertices = []
        for i, cell in enumerate(self.voronoi_vertices):
            pixCell = []
            for j, vertex in enumerate(cell):
                phyX, phyY = vertex
                pixX = phyX / xScalarRange * nX
                pixY = phyY / yScalarRange * nY
                pixCell.append((pixX, pixY))
            pixVertices.append(pixCell)

        # Now paint each of these cells with the colour of the nodeValue.
        # Use PIL for this?
        im = Image.new("F", (nX, nY))
        draw = ImageDraw.Draw(im)
        for i, cell in enumerate(pixVertices):
            draw.polygon(cell, fill=node_values[i])

        # Convert the PIL image to a np array
        # http://effbot.org/zone/pil-changes-116.htm
        imArray = np.asarray(im)  # will be read-only, so make a copy
        imArrayCopy = np.array(imArray, copy=True)

        return imArrayCopy

    def render_delaunay_field(self, node_values, x_range, y_range,
                              x_step, y_step, default=np.nan):
        """Renders a linearly interpolated Delaunay field.

        The Delaunay vertices take on the values of the nodes with linear
        interpolation across the triangular facets. Note that this field
        will not be continuously differentiable, but 'mass' will be conserved.

        Parameters
        ----------
        node_values : ndarray, `(nNodes,)`
            Array field values at each node
        x_range : tuple
            Tuple of (x_min, x_max)
        y_range : tuple
            Tuple of (y_min, y_max)
        x_step : float
            Scalar, size of pixels along x-axis
        y_step : float
            Scalar, size of pixels along y-axis
        default : scalar
            Value used outside the tessellation's convex hull

        Returns
        -------
        field : ndarray, (ny, nx)
            2D array (image) first-order interpolated Delaunay field.
        """
        interp = self.triangulation.linear_interpolator(node_values,
                                                        default_value=default)
        field = self._run_interpolator(interp,
                                       x_range, y_range,
                                       x_step, y_step)
        return field

    def render_nearest_neighbours_field(self, node_values, x_range, y_range,
                                        x_step, y_step, default=np.nan):
        """Renders a nearest-neighbours interpolated Delaunay Field.

        Nearest neighbours interpolation will create a continuously
        differentiable image, but 'mass' is not guaranteed to be conserved.

        Parameters
        ----------
        node_values : ndarray, `(nNodes,)`
            Array field values at each node
        x_range : tuple
            Tuple of (x_min, x_max)
        y_range : tuple
            Tuple of (y_min, y_max)
        x_step : float
            Scalar, size of pixels along x-axis
        y_step : float
            Scalar, size of pixels along y-axis
        default : scalar
            Value used outside the tessellation's convex hull

        Returns
        -------
        field : ndarray, (ny, nx)
            2D array (image) nearest-neighbours interpolated Delaunay field.
        """
        interp = self.triangulation.nn_interpolator(node_values,
                                                    default_value=default)
        field = self._run_interpolator(interp,
                                       x_range, y_range,
                                       x_step, y_step)
        return field

    def _run_interpolator(self, interp, x_range, y_range, x_step, y_step):
        """Runs Robert Kern's Linear or NN interpolator objects to create a
        field.
        """
        nX = int((x_range[1] - x_range[0]) / x_step)
        nY = int((y_range[1] - y_range[0]) / y_step)
        field = interp[y_range[0]:y_range[1]:complex(0, nY),
                       x_range[0]:x_range[1]:complex(0, nX)]
        return field
