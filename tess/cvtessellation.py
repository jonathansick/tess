#!/usr/bin/env python
# encoding: utf-8
"""
Centroidal Voronoi Tessellations (using Lloyd's algorithm).
"""

import numpy as np

from voronoi import VoronoiTessellation

from lloyd import lloyd


class CVTessellation(VoronoiTessellation):
    """Uses Lloyd's algorithm to assign data points to Voronoi bins so that
    each bin has an equal mass.

    Inherits from :class:`ghostmap.voronoi.VoronoiTessellation`.

    :param xPoints: array of cartesian `x` locations of each data point.
    :type xPoints: 1D `ndarray`
    :param yPoints: array of cartesian `y` locations of each data point.
    :type yPoints: 1D `ndarray`
    :param densPoints: Density *or weight* of each point. For an equal-S/N
                        generator, this should be set to (S/N)**2.
                        For an equal number generator this can be simple
                        an array of ones.
    :type densPoints: 1D `ndarray`
    :param preGenerator: an optional node generator already computed from
                            the data.
    :param useC: Set `False` to force use of pure-python Lloyd's algorithm
    """
    def __init__(self, xPoints, yPoints, densPoints, node_xy=None):
        xNode, yNode, vBinNum = self._tessellate(xPoints, yPoints, densPoints,
                node_xy=node_xy)
        super(CVTessellation, self).__init__(xNode, yNode)
        self.vBinNum = vBinNum
    
    def _tessellate(self, xPoints, yPoints, densPoints, node_xy=None):
        """ Computes the centroidal voronoi tessellation itself.

        :param xPoints: array of cartesian `x` locations of each data point.
        :type xPoints: 1D `ndarray`
        :param yPoints: array of cartesian `y` locations of each data point.
        :type yPoints: 1D `ndarray`
        :param densPoints: Density *or weight* of each point. For an equal-S/N
                           generator, this should be set to (S/N)**2.
                           For an equal number generator this can be simple
                           an array of ones.
        :type densPoints: 1D `ndarray`
        :param preGenerator: an optional node generator already computed from
                             the data.
        """
        self.densPoints = densPoints
        
        # Obtain pre-generator node coordinates
        if not node_xy is not None:
            node_xy = np.column_stack((xPoints.copy(), yPoints.copy()))

        xy = np.column_stack((xPoints, yPoints))
        node_xy, v_bin_numbers = lloyd(xy, densPoints, node_xy)
        return node_xy[:, 0], node_xy[:, 1], v_bin_numbers

    def _run_py_lloyds(self, xPoints, yPoints, densPoints, xNode, yNode):
        """Run Lloyd's algorithm in pure-python

        :param xPoints: array of cartesian `x` locations of each data point.
        :param yPoints: array of cartesian `y` locations of each data point.
        :param densPoints: array of the density of each point. For an equal-S/N
            generator, this should be set to (S/N)**2. For an equal number
            generator this can be simple an array of ones.
        :param xNode: array of cartesian `x` locations of each node.
        :param yNode: array of cartesian `y` locations of each node.
        """
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
                    xBar, yBar = self._weighted_centroid(xPoints[indices],
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
        
        :param x: array of x-axis spatial coordinates
        :param y: array of y-axis spatial coordiantes
        :param density: array containing the weighting values
        
        :return: tuple `(xBar, yBar)`, the weighted centroid
        """
        mass = np.sum(density)
        xBar = np.sum(x * density) / mass
        yBar = np.sum(y * density) / mass
        return (xBar, yBar)
    
    def get_node_membership(self):
        """Returns an array, the length of the input data arrays in
        `tessellate()`, which have indices into the node arrays of
        `get_nodes()`.
        """
        return self.vBinNum

    def get_node_weights(self):
        """Return the sum of the density for the nodes, same order as
        `get_nodes()`"""
        nNodes = len(self.xNode)
        nodeWeights = np.zeros(nNodes, dtype=np.float)
        for i in xrange(nNodes):
            ind = np.where(self.vBinNum == i)[0]
            if len(ind) > 0:
                nodeWeights[i] = np.sum(self.densPoints[ind])
        return nodeWeights
