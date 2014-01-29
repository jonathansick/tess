"""
Lloyd's Algorithm in Cython.
"""

from cython.view cimport array as cvarray
import numpy as np
from scipy.spatial import cKDTree


cpdef lloyd(double[:, :] xy, double[:] w, double[:, :] node_xy):
    cdef long i
    cdef long n_iters = 0
    cdef double delta, dx, dy
    cdef double [:, :] orig_node_xy
    cdef double [:] weight_sum

    cdef long n_nodes = node_xy.shape[0]
    cdef long n_points = xy.shape[0]
    cdef long [:] idx = np.zeros(n_points, dtype=int)  # voronoi bin indices
    cdef long [:] node_population = np.zeros(n_nodes, dtype=int)

    while True:
        # Copy the original nodes
        orig_node_xy = node_xy.copy()

        # Assign each point to the closest node
        # This defines a set of Voronoi bins
        # idx is length of xy, giving indices into node_xy
        tree = cKDTree(node_xy)
        idx = tree.query(xy, k=1)[1]

        # Compute weighted centroid of the Voronoi bins
        node_population = np.zeros(n_nodes, dtype=int)
        node_xy = np.zeros((n_nodes, 2), dtype=float)
        weight_sum = np.zeros(n_nodes, dtype=float)
        for i in xrange(n_points):
            node_population[idx[i]] += 1
            weight_sum[idx[i]] += w[i]
            for d in (0, 1):
                node_xy[idx[i], d] += w[i] * xy[i, d]
        for i in xrange(n_nodes):
            # print node_population[i]
            if node_population[i] > 0:
                node_xy[i, 0] /= weight_sum[i]
                node_xy[i, 1] /= weight_sum[i]
            else:
                # Empty node
                node_xy[i, 0] = 0.
                node_xy[i, 1] = 0.

        # Compute how much each node has moved
        delta = 0.
        for i in xrange(n_nodes):
            dx = orig_node_xy[i, 0] - node_xy[i, 0]
            dy = orig_node_xy[i, 1] - node_xy[i, 1]
            delta += dx * dx + dy * dy
        print "Delta %03d %.2e" % (n_iters, delta)

        # Judge convergence
        if delta == 0:
            return node_xy, idx
        elif n_iters > 300:
            return None
        else:
            n_iters += 1
