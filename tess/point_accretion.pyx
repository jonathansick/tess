"""
Cython point accretion module -- for binning points until a threshold mass or
S/N is met.
"""
from cython.view cimport array as cvarray
import numpy as np
from scipy.spatial import cKDTree

cdef extern from "math.h":
    double sqrt(double x)


cdef class PointAccretor:
    cdef double [:, :] xy
    cdef double [:] w
    cdef long [:] bin_nums  # bin ID of each point
    cdef long [:] good_bin  # 1 if a well-made bin
    cdef long _n_unbinned  # count unbinned points
    cdef long n_bins  # number of bins
    cdef object tree  # KD Tree of points

    cpdef accrete(self):
        """Run the point accretion algorithm to build bins of points of a
        certain mass or quality.
        """
        cdef long [:] current_bin  # indices of items in current bin
        cdef double [:] current_bin_xy  # centroid of current bin
        cdef long current_bin_count = 0  # number of items in current bin
        self.n_bins = 0
        self._n_unbinned = self.xy.shape[0]  # count unbinned points
        self.bin_nums = np.zeros(self.xy.shape[0], dtype=int)
        self.good_bin = np.zeros(self.xy.shape[0], dtype=int)

        # Seed position is centroid of distribution
        xyc = self.centroid(np.arange(0, self._n_unbinned), self._n_unbinned)

        # Build a KDTree of all points
        self.tree = cKDTree(self.xy)

        while self._n_unbinned > 0:
            # Initialize bin
            self.n_bins += 1
            idx = self.tree.query(xyc, k=1)[1]
            current_bin = np.zeros(self._n_unbinned, dtype=int)
            current_bin[0] = idx
            self.bin_nums[<long>idx] = self.n_bins  # use n_bins as a bin ID
            current_bin_count = 1
            self._n_unbinned -= 1
            xyc[0] = self.xy[idx, 0]
            xyc[1] = self.xy[idx, 1]
            
            # Accrete points
            while not self.is_bin_full(current_bin, current_bin_count) \
                    and self._n_unbinned > 0:
                idx = self.find_closest_unbinned(xyc, current_bin_count)
                # Add this point to the bin
                current_bin_count += 1
                self._n_unbinned -= 1
                current_bin[current_bin_count - 1] = idx
                self.bin_nums[idx] = self.n_bins
                xyc = self.centroid(current_bin, current_bin_count)

            # Check if the bin is complete
            if self.is_bin_full(current_bin, current_bin_count):
                self.good_bin[self.n_bins - 1] = 1

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

    cdef centroid(self, long [:] inds, long n_points):
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

    cdef long find_closest_unbinned(self, double [:] xyc, long n_binned):
        cdef int keep_going
        cdef long idx, i, j
        # n is number of points find around node; hopefully enough so find
        # unbinned points
        cdef long n = 10 + n_binned
        if n > self.xy.shape[0]:
            n = self.xy.shape[0]
        keep_going = 1
        while keep_going:
            if n == self.xy.shape[0]:
                keep_going = 0
            indices = self.tree.query(xyc, k=n)[1]
            # Find first index not binned already
            # implicity assumes indices is sorted by distance for xyc
            for i in xrange(n):
                idx = indices[i]
                if self.bin_nums[idx] > 0:
                    continue
                else:
                    return idx
            # If here, no matches; expand number of points to return
            n = 10 + n
            if n > self.xy.shape[0]:
                n = self.xy.shape[0]

    cpdef cleanup(self):
        """Clean up bins that failed to meet quality requirements by
        re-allocating their points to other bins.
        """
        cdef long i, j, current_bin_count, n_good_bins, current_bin_num
        cdef long [:] current_bin  # indices of items in current bin
        i = 0
        n_good_bins = 0
        while i < self.n_bins:
            if self.good_bin[i]:
                n_good_bins += 1
            else:
                # Need to re-allocate this bin.
                # List all points in this bin
                current_bin_count = 0
                current_bin = np.zeros(self.xy.shape[0], dtype=int)
                current_bin_num = i + 1
                for j in xrange(self.xy.shape[0]):
                    if self.bin_nums[j] == current_bin_num:
                        current_bin[current_bin_count] = j
                        current_bin_count += 1
                # Redistribute the points
                self._redistribute_points(i, current_bin, current_bin_count)
                i -= 1  # since we removed a bin
            i += 1

    cpdef _redistribute_points(self, long bin_index, long [:] current_bin,
            long current_bin_count):
        cdef long j, k, old_bin_num
        cdef double [:, :] node_xy = self.nodes()  # node centroids
        cdef double [:] xyc = np.empty(2, dtype=float)
        node_tree = cKDTree(node_xy)
        for j in xrange(current_bin_count):
            k = current_bin[j]
            # re-allocate point
            xyc[0] = self.xy[k, 0]
            xyc[1] = self.xy[k, 1]
            indices = node_tree.query(xyc, k=2)[1]
            if indices[0] == bin_index:
                # Use the other index instead; don't allocate back to itself
                self.bin_nums[j] = indices[1] + 1  # since bin_nums is 1-based
            else:
                # First index is different+okay
                self.bin_nums[j] = indices[0] + 1  # since bin_nums is 1-based

        # Re-number all bins that come after bin_index
        old_bin_num = bin_index + 1
        for j in xrange(self.xy.shape[0]):
            if self.bin_nums[j] >= old_bin_num:
                self.bin_nums[j] -= 1
        self.n_bins -= 1


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


cdef class EqualSNAccretor(PointAccretor):
    """Handles point accretion so each bin has roughly equal S/N.
    
    Bin centroids will be weighted by point signal.
    """
    cdef double target_sn
    cdef double [:] variance

    def __init__(self, double [:, :] xy, double [:] signal, double [:] noise,
            double target_sn):
        cdef long i
        self.target_sn = target_sn
        self.xy = xy
        self.w = signal
        self.variance = np.empty(signal.shape[0], dtype=float)
        for i in xrange(signal.shape[0]):
            self.variance[i] = noise[i] * noise[i]

    cpdef is_bin_full(self, long [:] current_bin, long n):
        cdef double total_variance = 0.
        cdef double total_signal = 0.
        cdef double total_sn = 0.
        for i in xrange(n):
            total_signal += self.w[current_bin[i]]
            total_variance += self.variance[current_bin[i]]
        total_sn = total_signal / sqrt(total_variance) 
        if total_sn >= self.target_sn:
            return True
        else:
            return False
