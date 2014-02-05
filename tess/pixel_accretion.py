#!/usr/bin/env python
# encoding: utf-8
"""
Pixel accretion methods for segmenting images.
"""
from heapq import heappop, heapify, heappush

import numpy as np


class PixelAccretor(object):
    """Baseclass for pixel accretion.

    Unlike the point accretion algorithms in :mod:`tess.point_accretion`,
    these pixel accretion tools uses the grid geometry of images and only
    accrete pixels are connected neighbours.
    
    Users are expected to build or use a subclass of :class:`PixelAccretor`
    that need to implement the following methods

    - ``bin_started`` when a new bin is seeded.
    - ``candidate_quality``, give quality value for pixel w.r.t bin.
    - ``accept_pixel``, True if pixel should be accepted.
    - ``pixel_added``, called when a pixel is accepted.
    - ``close_bin``, called when a bin is completed.

    See :class:`tess.pixel_accretion.IsoIntensityAccrector` for an example.
    """
    def __init__(self):
        super(PixelAccretor, self).__init__()

    def accrete(self, ij0):
        """Run the pixel accretion algorithm, starting with pixel ij0.
        
        Parameters
        ----------
        idx : tuple
            Index of first pixel.
        """
        # self.n_unbinned_pixels = self.image.shape[0] * self.image.shape[1]
        self._global_edge_pixels = set([])
        self._seg_image = -1 * np.ones(self.image.shape, dtype=int)
        self._nrows, self._ncols = self.image.shape
        n_bins = 0
        while ij0:
            self._make_bin(ij0, n_bins)
            ij0 = self._new_start_point()
            n_bins += 1

    def _make_bin(self, ij0, bin_index):
        """Make a new bin, starting with pixel ij0."""
        self.current_bin_indices = [ij0]
        self.current_edge_heap = None
        self.current_edge_dict = {}  # to keep index into heap
        self._seg_image[ij0] = bin_index
        self._add_edges(ij0)
        self.bin_started()  # call to subclass
        while self.current_edge_heap:  # while there are edges
            # Select a new pixel to add
            quality, ij0 = heappop(self.current_edge_heap)
            if self.accept_pixel(ij0):
                # Add pixel
                del self.current_edge_dict[ij0]
                self.current_bin_indices.append(ij0)
                self._seg_image[ij0] = bin_index
                self._add_edges(ij0)
                self.pixel_added()  # call to subclass
            else:
                # Reject pixel and stop accretion
                self.close_bin()  # call to subclass
                break
        # Add remaining edges to the global edge list
        leftovers = set(self.current_edge_dict.keys())
        # Remove accreted points from global edges
        self._global_edge_pixels -= set(self.current_bin_indices)
        self._global_edge_pixels = self._global_edge_pixels.union(leftovers)

    def _add_edges(self, ij0):
        """Add edges surrounding ij0 that aren't binned already. As edges are
        added, the super class is asked to make a scalar judgement of the
        desireability of this pixel."""
        # Top neighbour
        idx = (ij0[0] + 1, ij0[1])
        if idx[0] < self._nrows and self._seg_image[idx] == -1:
            if not idx in self.current_edge_dict:
                quality = self.candidate_quality(idx)  # call to subclass
                v = (quality, idx)
                self.current_edge_dict[idx] = v
                if self.current_edge_heap:
                    heappush(self.current_edge_heap, v)
                else:
                    self.current_edge_heap = [v]
                    heapify(self.current_edge_heap)

        # Bottom neighbour
        idx = (ij0[0] - 1, ij0[1])
        if idx[0] >= 0 and self._seg_image[idx] == -1:
            if not idx in self.current_edge_dict:
                quality = self.candidate_quality(idx)  # call to subclass
                v = (quality, idx)
                self.current_edge_dict[idx] = v
                if self.current_edge_heap:
                    heappush(self.current_edge_heap, v)
                else:
                    self.current_edge_heap = [v]
                    heapify(self.current_edge_heap)

        # Right neighbour
        idx = (ij0[0], ij0[1] + 1)
        if idx[1] < self._ncols and self._seg_image[idx] == -1:
            if not idx in self.current_edge_dict:
                quality = self.candidate_quality(idx)  # call to subclass
                v = (quality, idx)
                self.current_edge_dict[idx] = v
                if self.current_edge_heap:
                    heappush(self.current_edge_heap, v)
                else:
                    self.current_edge_heap = [v]
                    heapify(self.current_edge_heap)

        # Left neighbour
        idx = (ij0[0], ij0[1] - 1)
        if idx[1] >= 0 and self._seg_image[idx] == -1:
            if not idx in self.current_edge_dict:
                quality = self.candidate_quality(idx)  # call to subclass
                v = (quality, idx)
                self.current_edge_dict[idx] = v
                if self.current_edge_heap:
                    heappush(self.current_edge_heap, v)
                else:
                    self.current_edge_heap = [v]
                    heapify(self.current_edge_heap)

    def update_edge_heap(self):
        """May be called by the subclass whenever the 'quality' values of edge
        pixels need to be recomputed because the bin itself has changed
        significantly.

        *It is up to the subclass to call this method as necessary*. Calling
        this method infrequently will speed up the pixel accretion, but may
        cause suboptimal choices of pixels being accreted into bins.
        """
        self.current_edge_heap = []
        for idx, v in self.current_edge_dict.iteritems():
            new_q = self.candidate_quality(idx)
            new_v = (new_q, idx)
            self.current_edge_dict[idx] = new_v
            self.current_edge_heap.append(new_v)
        heapify(self.current_edge_heap)

    def _new_start_point(self):
        """Suggest a new starting pixel for next bin.
        
        This pixel comes from the pool of edge pixels.
        Returns ``None`` if no edge pixels are available.
        """
        try:
            ij_next = self._global_edge_pixels.pop()
        except KeyError:
            return None
        return ij_next


class IsoIntensityAccretor(PixelAccretor):
    """Bin pixels to make iso-intensity regions.

    Each region in the image will have a standard deviation of pixel
    intensities within a user-defined limit. This can be thought of as
    a way of building non-parameteric isophotal regions.
    
    Parameters
    ----------
    image : ndarray
        The image to be segmented.
    intensity_sigma_limit : float
        Maximum standard deviation of pixel intensities permitted in a
        single bin.
    min_pixels : int
        Minimum number of pixels that need to be in a single bin.
    max_pixels : int
        Maximum number of pixels that can be accreted into a single bin.
        If ``None``, then no limit is enforced.
    max_shift_frac : flaot
        Maximum fractional change of the bin's mean before the edge pixel
        heap is updated.
    """
    def __init__(self, image, intensity_sigma_limit,
            min_pixels=1, max_pixels=None, max_shift_frac=0.05):
        super(IsoIntensityAccretor, self).__init__()
        self.image = image
        self.intensity_sigma_limit = intensity_sigma_limit
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self._bin_mean_intensity = None
        self._max_shift_frac = max_shift_frac

    def _update_bin_mean_intensity(self):
        """Compute self._bin_mean_intensity."""
        mu = sum([self.image[idx] for idx in self.current_bin_indices]) \
                / float(len(self.current_bin_indices))
        self._bin_mean_intensity = mu

    def bin_started(self):
        """Called by :class`PixelAccretor` baseclass when a new bin has been
        started (and a seed pixel has been added)."""
        self._update_bin_mean_intensity()
        # Since this is the first time mean_shift_intensity is added,
        # hold onto the original value
        self._original_bin_mean_intensity = self._bin_mean_intensity

    def candidate_quality(self, idx):
        """Gives the scalar quality of adding this pixel with respect to the
        current bin. Pixels with the smallest 'quality' value are accreted.
        Here quality is defined as absolute difference of the pixel's intensity
        and the mean intensity of the existing bin.
        
        Parameters
        ----------
        idx : tuple
            The pixel index to be tested.
        """
        return float(np.abs(self.image[idx] - self._bin_mean_intensity))

    def accept_pixel(self, idx):
        """Test a pixel, return ``True`` if it should be added to the bin.
        
        Parameters
        ----------
        idx : tuple
            The pixel index to be tested..
        """
        npix = len(self.current_bin_indices)
        if npix < self.min_pixels:
            return True
        if self.max_pixels and npix > self.max_pixels:
            return False
        intensities = np.array([self.image[k] for k in
            self.current_bin_indices + [idx]])
        if intensities.std() > self.intensity_sigma_limit:
            return False
        else:
            return True

    def pixel_added(self):
        """Called once a pixel has been added."""
        self._update_bin_mean_intensity()
        frac_diff = (self._original_bin_mean_intensity \
            - self._bin_mean_intensity) / self._original_bin_mean_intensity
        if np.abs(frac_diff) > self._max_shift_frac:
            # Update the edge heap if the mean has shifted by more than 5%.
            self.update_edge_heap()
            # Update definition of original bin intensity
            self._original_bin_mean_intensity = self._bin_mean_intensity

    def close_bin(self):
        """Called when the current bin is completed."""
        self._bin_mean_intensity = None
