"""
Cython point accretion module -- for binning points until a threshold mass or
S/N is met.
"""
from cython.view cimport array as cvarray
import numpy as np
from scipy.spatial import cKDTree


cdef class PointAccretor:
    cdef double [:, :] xy
    cdef double [:] w
    cdef long [:] bin_nums  # bin ID of each point
    cdef long _n_unbinned  # count unbinned points
    cdef long n_bins  # number of bins

    cpdef accrete(self):
        cdef long [:] current_bin  # indices of items in current bin
        cdef double [:] current_bin_xy  # centroid of current bin
        cdef long current_bin_count = 0  # number of items in current bin
        self.n_bins = 0
        self._n_unbinned = self.xy.shape[0]  # count unbinned points
        self.bin_nums = np.zeros(self.xy.shape[0], dtype=int)

        # Seed position is centroid of distribution
        xyc = self.centroid(np.arange(0, self._n_unbinned), self._n_unbinned)

        while self._n_unbinned > 0:
            # Initialize bin
            self.n_bins += 1
            idx = self.find_closest_unbinned(xyc)
            current_bin = np.zeros(self._n_unbinned, dtype=int)
            current_bin[0] = idx
            self.bin_nums[idx] = self.n_bins  # use n_bins as a bin ID
            current_bin_count = 1
            self._n_unbinned -= 1
            xyc[0] = self.xy[idx, 0]
            xyc[1] = self.xy[idx, 1]
            
            while not self.is_bin_full(current_bin, current_bin_count) \
                    and self._n_unbinned > 0:
                idx = self.find_closest_unbinned(xyc)
                # Add this point to the bin
                current_bin_count += 1
                self._n_unbinned -= 1
                current_bin[current_bin_count - 1] = idx
                self.bin_nums[idx] = self.n_bins
                xyc = self.centroid(current_bin, current_bin_count)

    cpdef nodes(self):
        """Return the x,y coordinates of the node centroids."""
        cdef long i, j
        cdef double [:, :] node_xy = np.zeros((self.n_bins, 2), dtype=float)
        cdef double [:] node_m = np.zeros(self.n_bins, dtype=float)
        print self.n_bins
        for i in xrange(self.xy.shape[0]):
            j = self.bin_nums[i] - 1  # for zero-based index
            node_xy[j, 0] += self.xy[i, 0] * self.w[i]
            node_xy[j, 1] += self.xy[i, 1] * self.w[i]
            node_m[j] += self.w[i]
        for j in xrange(self.n_bins):
            node_xy[j, 0] /= node_m[j]
            node_xy[j, 1] /= node_m[j]
        return node_xy

    cpdef centroid(self, long [:] inds, long n_points):
        """Compute centroid of points given by the index array ``inds``."""
        cdef double [:] xyc = np.zeros(2, dtype=float)
        cdef double mass_sum = 0
        for i in xrange(n_points):
            xyc[0] += self.xy[inds[i], 0] * self.w[inds[i]]
            xyc[1] += self.xy[inds[i], 1] * self.w[inds[i]]
            mass_sum += self.w[inds[i]]
        xyc[0] /= mass_sum
        xyc[1] /= mass_sum
        return xyc

    cpdef long find_closest_unbinned(self, xyc):
        # Build coordinates of all unbinned points
        cdef double [:, :] uxy = np.empty((self._n_unbinned, 2), dtype=float)
        cdef long [:] orig_id = np.empty(self._n_unbinned, dtype=int)
        cdef long j = 0
        for i in xrange(self.xy.shape[0]):
            if self.bin_nums[i] == 0:  # is unbinned
                uxy[j, 0] = self.xy[i, 0]
                uxy[j, 1] = self.xy[i, 1]
                orig_id[j] = i
                j += 1

        # Use a KD tree to find point closest to xc
        tree = cKDTree(uxy)
        idx = tree.query(xyc, k=1)[1]
        return orig_id[<long>idx]


cdef class EqualMassAccretor(PointAccretor):
    """Handles point accretion so each bin has roughly equal mass."""
    cdef double target_mass

    def __init__(self, double [:, :] xy, double [:] mass, double target_mass):
        self.target_mass = target_mass
        self.xy = xy
        self.w = mass

    cpdef is_bin_full(self, long [:] current_bin, long n):
        cdef double total_mass = 0.
        for i in xrange(n):
            total_mass += self.w[current_bin[i]]
        if total_mass >= self.target_mass:
            return True
        else:
            return False
