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
    """Creates a Delaunay triangulation of the given nodes."""
    def __init__(self, xNode, yNode):
        super(DelaunayTessellation, self).__init__()
        self.xNode = xNode
        self.yNode = yNode
        self.triangulation = Triangulation(self.xNode, self.yNode)
        self.memTable = None # membership table. Can be computed with _computeMembershipTable()
    
    def get_triangulation(self):
        """:return: the Triangulation object (from Kern's package)"""
        return self.triangulation
    
    def get_nodes(self):
        """:return: tuple of (x, y), coordinates of tessellation nodes."""
        return (self.xNode, self.yNode)
    
    def get_extreme_nodes(self):
        """Extreme nodes are those on the convex hull of the whole node distribution.
        :return: np array (n, 1) listing the node indices that make up the
            convex hull. Ordered counter-clockwise."""
        return self.triangulation.hull
    
    def get_triangles(self):
        """:return: np array (nTri by 3) of node indices of each triangle."""
        return self.triangulation.triangle_nodes
    
    def get_adjacency_matrix(self):
        """Gets matrix describing the neighbouring triangles of each triangle.
        
        .. note:: As per Robert Kern's Delaunay package: The value can also be -1 meaning
            that that edge is on the convex hull of
            the points and there is no neighbor on that edge. The values are ordered
            such that triangle_neighbors[tri, i] corresponds with the edge
            *opposite* triangle_nodes[tri, i]. As such, these neighbors are also in
            counter-clockwise order.
        
        :return: np array (nTri by 3) indices of triangles that share a
            side with each triangle. Same order as the triangle array.
        """
        return self.triangulation.triangle_neighbors
    
    def get_circumcenters(self):
        """Gets an array of the circumcenter of each Delaunay triangle.
        :return: np array (nTri, 2), whose values are the (x,y) location
            of the circumcenter. Ordered like other triangle arrays.
        """
        return self.triangulation.circumcenters
    
    def get_membership_table(self):
        """The membership table gives the indices of all triangles associated
        with a given node.
        :return: A list nNodes long (in same order as other node arrays). Each
            item is another list giving the triangle indices.
        """
        if self.memTable is None:
            self._make_membership_table()
        return self.memTable
    
    def get_number_of_triangles(self):
        """:return: integer number of triangles in Delaunay triangulation."""
        return self.triangulation.triangle_nodes.shape[0]
    
    def compute_triangle_areas(self):
        """Computes the geometric area of each triangle.
        :return: (nTri, 1) np array of triangle areas.
        """
        print "Computing triangle areas"
        # Make abbreviated pointers
        tri = self.get_triangles()
        x = self.xNode
        y = self.yNode
        # Compute side lengths
        a = np.sqrt((x[tri[:,1]]-x[tri[:,0]])**2. + (y[tri[:,1]]-y[tri[:,0]])**2.) # between verts 1 and 0
        b = np.sqrt((x[tri[:,2]]-x[tri[:,1]])**2. + (y[tri[:,2]]-y[tri[:,1]])**2.) # between verts 2 and 1
        c = np.sqrt((x[tri[:,2]]-x[tri[:,0]])**2. + (y[tri[:,2]]-y[tri[:,0]])**2.) # between verts 2 and 0
        s = (a+b+c) / 2. # semi-perimeter
        areas = np.sqrt(s*(s-a)*(s-b)*(s-c)) # Heron's formula
        return areas
    
    def compute_voronoi_cell_vertices(self):
        """Computes the polygon vertices of the Voronoi cells surrounding each node."""
        # Function to assess the radial location of a point
        # Based on http://stackoverflow.com/questions/1709283/how-can-i-sort-a-coordinate-list-for-a-rectangle-counterclockwise
        ccwDist = lambda xy: (math.atan2(xy[1] - y0, xy[0] - x0) + 2.*math.pi) % 2.*math.pi
        
        xNode, yNode = self.get_nodes()
        memTable = self.get_membership_table()
        circumcenters = self.get_circumcenters()
        extremeNodes = self.get_extreme_nodes()
        nExtremeNodes = len(extremeNodes)
        print extremeNodes
        cells = []
        for i, members in enumerate(memTable):
            # voronoi vertices are the circumcenters of triangles the node belongs to
            voronoiCell = [circumcenters[j] for j in members]
            # if the node is on the convext hull... then add as points the bisections of
            # of the neighbouring hull edges and the node itself.
            if i in extremeNodes:
                print "%i is on hull; there are %i nodes on hull" % (i, nExtremeNodes)
                # find the neighbouring extreme points, the one to the left and right
                # of the point being studied
                k = extremeNodes.index(i) # index into extreme nodes list
                if k > 0:
                    rightNode = extremeNodes[k-1]
                else:
                    rightNode = extremeNodes[nExtremeNodes-1] # take the wrap-around
                if k < nExtremeNodes-1:
                    leftNode = extremeNodes[k+1]
                else:
                    leftNode = extremeNodes[0]
                # Now left and right nodes are indices into the nodes list
                # of the node's hull neighbours
                print "\tneighbours: %i and %i" % (leftNode, rightNode)
                voronoiCell.append(((xNode[rightNode]+xNode[i])/2.,(yNode[rightNode]+yNode[i])/2.))
                voronoiCell.append(((xNode[leftNode]+xNode[i])/2.,(yNode[leftNode]+yNode[i])/2.))
                voronoiCell.append((xNode[i],yNode[i]))
            # Find the geometric centre of the cell
            nP = len(voronoiCell) # number of vertices of the voronoi polygon
            x0 = sum(p[0] for p in voronoiCell) / nP
            y0 = sum(p[1] for p in voronoiCell) / nP
            voronoiCell.sort(key=ccwDist) # ensure the vertices are sorted CCW
            cells.append(voronoiCell)
        
        return cells
    
    def _make_membership_table(self):
        """Computes the membership table of the triangles that each node is
        associated with."""
        xNode, yNode = self.get_nodes()
        nNodes = len(xNode)
        self.memTable = [None]*nNodes
        nTri = self.get_number_of_triangles()
        triangleMatrix = self.get_triangles()
        vertices = range(3)
        for i in xrange(nTri):
            for j in vertices:
                theNode = triangleMatrix[i,j]
                if self.memTable[theNode] == None:
                    self.memTable[theNode] = [i]
                else:
                    # check to see if the triangle is noted for theNode yet
                    triExists = False
                    for k in range(len(self.memTable[theNode])):
                        if i == self.memTable[theNode][k]:
                            triExists = True
                    if triExists == False:
                        self.memTable[theNode].append(i)


class VoronoiDensityEstimator(object):
    """Uses a DelaunayTessellation and point data to set the density of each
    Voronoi cell. This can then be used as the values for the zeroth-order
    Voronoi density field reconstruction.
    """
    def __init__(self):
        super(VoronoiDensityEstimator, self).__init__()


class DelaunayDensityEstimator(object):
    """Uses the DTFE to set the density of each Delaunay node from a
    DelaunayTessellation and the point data. This can then be used as the values
    for the first-order Delaunay density field reconstruction.
    
    :param delaunayTessellation: A `DelaunayTessellation` object, which has the
        delaunay triangulation
    """
    def __init__(self, delaunayTessellation):
        super(DelaunayDensityEstimator, self).__init__()
        self.delaunayTessellation = delaunayTessellation
        self.nodeDensity = None
        # hullCorrection: if True, then correct the incomplete area around
        # Delaunay vertices on point distribution's hull
        # NOTE: this seems to be broken
        self.hullCorrection = False
    
    def estimate_density(self, xRange, yRange, nodeMasses, pixelMask=None):
        """Estimate the density of each node in the delaunay tessellation using
        the DTFE of Schaap (PhD Thesis, Groningen).
        :param xRange: a tuple of (x_min, x_max); TODO make these optional, just
            to scale the pixel mask array
        :param yRange: a tuple of (y_min, y_max)
        :param nodeMasses: an array of the mass of each node, in the same order
            used by the `delaunayTessellation` object
        :param pixelMask: np array of same size as the extent of `xRange`
            and `yRange`. A pixel value of 1 is masked, values of 0 are admitted.
            This allows masked areas to be excluded from the density computation.
        """
        xNode, yNode = self.delaunayTessellation.get_nodes()
        nNodes = len(xNode)
        if nNodes != len(nodeMasses):
            print "Warning: nodeMasses has wrong length (%i, should be %i)" % \
                (len(xNode), len(nodeMasses))
        
        #triangles = self.delaunayTessellation.get_triangles()
        #adjMatrix = self.delaunayTessellation.get_adjacency_matrix()
        memTable = self.delaunayTessellation.get_membership_table()
        nTri = self.delaunayTessellation.get_number_of_triangles()
        areas = self.delaunayTessellation.compute_triangle_areas()
        extremeNodes = self.delaunayTessellation.get_extreme_nodes()
        
        # TODO this is where I would compute the masked areas...
        # use the Walking Triangle algorithm
        maskedAreas = np.zeros(nTri)
        if pixelMask is not None:
            pass
        
        # Compute the area of triangles contiguous about each node
        contigAreas = np.zeros([nNodes])
        for i in xrange(nNodes):
            contigTris = memTable[i] # list of triangle indices associated with node i
            contigAreas[i] = areas[contigTris].sum() - maskedAreas[contigTris].sum()
        
        # Correct the area of contiguous Voronoi regions on the outside of the
        # convex hull.
        # TODO check this
        if self.hullCorrection:
            nExtremeNodes = len(extremeNodes)
            for i, node in enumerate(extremeNodes):
                # find the neighbouring extreme points, the one to the left and right
                # of the point being studied
                if i > 0:
                    rightNode = extremeNodes[i-1]
                else:
                    rightNode = extremeNodes[nExtremeNodes-1] # take the wrap-around
                if i < nExtremeNodes-1:
                    leftNode = extremeNodes[i+1]
                else:
                    leftNode = extremeNodes[0]
                # find the angle that they subtend, using The Law of Cosines
                a = math.sqrt((xNode[leftNode]-xNode[node])**2 + (yNode[leftNode]-yNode[node])**2)
                b = math.sqrt((xNode[rightNode]-xNode[node])**2 + (yNode[rightNode]-yNode[node])**2)
                c = math.sqrt((xNode[rightNode]-xNode[leftNode])**2 + (yNode[rightNode]-yNode[node])**2)
                subAngle = math.acos((a**2+b**2-c**2)/(2*a*b))
                # The sub angle should in theory be less than 180 degrees. This code
                # ensures that we have an angle that covers more than 180 degrees.
                extraAngle = 2*math.pi - subAngle
                if extraAngle < math.pi:
                    print "Angle error in edge effect correction"
                correctionFactor = extraAngle / subAngle
                # update the contiguous area:
                contigAreas[node] = (1.+correctionFactor)*contigAreas[node]
        
        # Finally compute the density at the site of each node by using
        # eqn 3.36 of Schaap 2007 (pg 69)
        self.nodeDensity = 3. * nodeMasses / contigAreas
        
        # Compute the total tessellated area
        self.totalArea = contigAreas.sum() / 3.
        
        return self.nodeDensity


class FieldRenderer(object):
    """Renders the Delaunay-tessellated field to a pixel-based image."""
    def __init__(self, delaunayTessellation):
        super(FieldRenderer, self).__init__()
        self.delaunayTessellation = delaunayTessellation
    
    def render_zeroth_order_voronoi(self, nodeValues, xRange, yRange, xStep, yStep):
        """Renders a zeroth-order field (a Voronoi tiling).
        :param nodeValues: np array (nNodes, 1) of the field values at each node
        :param xRange: tuple of (x_min, x_max)
        :param yRange: tuple of (y_min, y_max)
        :param xStep: scalar, size of pixels along x-axis
        :param yStep: scalar, size of pixels along y-axis
        """
        cellVertices = self.delaunayTessellation.compute_voronoi_cell_vertices()
        
        xScalarRange = xRange[1]-xRange[0]
        yScalarRange = yRange[1]-yRange[0]
        
        nX = int(xScalarRange / xStep)
        nY = int(yScalarRange / yStep)
        
        # Transform the cell vertices in physical units to the pixel units of
        # the rendering space
        pixVertices = [] # same as `cellVertices` but in pixel space
        for i, cell in enumerate(cellVertices):
            pixCell = []
            for j, vertex in enumerate(cell):
                phyX, phyY = vertex
                pixX = phyX / xScalarRange * nX
                pixY = phyY / yScalarRange * nY
                pixCell.append((pixX,pixY))
            pixVertices.append(pixCell)
        
        # Now paint each of these cells with the colour of the nodeValue.
        # Use PIL for this?
        im = Image.new("F", (nX,nY))
        draw = ImageDraw.Draw(im)
        for i, cell in enumerate(pixVertices):
            draw.polygon(cell, fill=nodeValues[i])
        
        # Convert the PIL image to a np array
        # http://effbot.org/zone/pil-changes-116.htm
        imArray = np.asarray(im) # will be read-only, so make a copy
        imArrayCopy = np.array(imArray, copy=True)
        
        return imArrayCopy
    
    def render_first_order_delaunay(self, nodeValues, xRange, yRange,
            xStep, yStep, defaultValue=np.nan):
        """Renders a linearly interpolated Delaunay field.
        :param nodeValues: np array (nNodes, 1) of the field values at each node
        :param xRange: tuple of (x_min, x_max)
        :param yRange: tuple of (y_min, y_max)
        :param xStep: scalar, size of pixels along x-axis
        :param yStep: scalar, size of pixels along y-axis
        :param defaultValue: scalar value used outside the tessellation's convex hull
        """
        interp = self.delaunayTessellation.get_triangulation().linear_interpolator(
                nodeValues, default_value=defaultValue)
        field = self._run_interpolator(interp, xRange, yRange, xStep, yStep)
        return field
    
    def render_nearest_neighbours_delaunay(self, nodeValues, xRange, yRange,
            xStep, yStep, defaultValue=np.nan):
        """docstring for renderNearestNeighboursDelaunay.
        :param nodeValues: np array (nNodes, 1) of the field values at each node
        :param xRange: tuple of (x_min, x_max)
        :param yRange: tuple of (y_min, y_max)
        :param xStep: scalar, size of pixels along x-axis
        :param yStep: scalar, size of pixels along y-axis
        :param defaultValue: scalar value used outside the tessellation's convex hull
        """
        interp = self.delaunayTessellation.get_triangulation().nn_interpolator(self,
                nodeValues) # , default_value=defaultValue
        field = self._run_interpolator(interp, xRange, yRange, xStep, yStep)
        return field
    
    def _run_interpolator(self, interp, xRange, yRange, xStep, yStep):
        """Runs Robert Kern's Linear or NN interpolator objects to create a field."""
        nX = int((xRange[1]-xRange[0])/xStep)
        nY = int((yRange[1]-yRange[0])/yStep)
        field = interp[yRange[0]:yRange[1]:complex(0,nY),
                    xRange[0]:xRange[1]:complex(0,nX)]
        return field

def makeRectangularBinnedDensityField(x, y, mass, xRange, yRange, xBinSize, yBinSize):
    """Does rectangular binning on a point distribution. Returns a field with
    pixels in units of sum(mass_points)/area. Each pixel in the field is a bin.
    If you want a bin to occupy more than one pixel, just resize the np array.
    
    :param x: np array of x point coordinates
    :param y: np array of y point coordinates
    :param mass: np array of point masses
    :param xRange: tuple of (xmin, xmax); determines range of reconstructed field
    :param yRange: tuple of (ymin, ymax)
    :param xBinSize: scalar, length of bins along x-axis
    :param yBinSize: scalar, length of bins along y-axis
    """
    xGrid = np.arange(min(xRange), max(xRange)-xBinSize, xBinSize)
    yGrid = np.arange(min(yRange), max(yRange)-yBinSize, yBinSize)
    field = np.zeros([len(yGrid),len(xGrid)])
    binArea = xBinSize * yBinSize
    
    # Trim the dataset to ensure it only covers the range of the field reconstruction
    good = np.where((x>min(xRange)) & (x<max(xRange))
        & (y>min(yRange)) & (y<max(yRange)))[0]
    x = x[good]
    y = y[good]
    mass = mass[good]
    
    # Then sort the point into increasing x-coordiante
    xsort = np.argsort(x)
    x = x[xsort]
    y = y[xsort]
    mass = mass[xsort]
    
    for xi, xg in enumerate(xGrid):
        col = np.where((x>xg) & (x<(xg+xBinSize)))[0]
        if len(col) == 0: continue # no points in whole column
        # xcol = x[col]
        ycol = y[col]
        mcol = mass[col]
        for yi, yg in enumerate(yGrid):
            bin = np.where((ycol>yg) & (ycol<(yg+yBinSize)))[0]
            if len(bin) == 0: continue # no points in bin
            totalMass = np.sum(mcol[bin])
            field[yi,xi] = totalMass
    
    # Make it a density plot by dividing each pixel by the binArea
    field = field / binArea
    
    return field
