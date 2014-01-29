#!/usr/bin/env python
# encoding: utf-8
"""
For deriving the projected stellar density a resolved stellar population,
given colour information and a spatial distribution.

History
-------
2011-10-18 - Created by Jonathan Sick
"""

import numpy as np
import pyfits
import matplotlib as mpl
import matplotlib.pyplot as plt

import ghostmap
from cvtessellation import CVTessellation


class StarField(object):
    """Class for storing stellar data and driving the density estimation
    pipeline.
    
    This class could be inherited and overriden to support specific data
    sources, or data can be loaded using the class method
    :meth:`load_arrays`.
    """
    def __init__(self):
        super(StarField, self).__init__()
        self.x = None
        self.y = None
        self.mag1 = None
        self.mag2 = None
        self.weight = None
        self.selection = None
        self.polySelection = None
        self.wcs = None
        self.nStars = 0
        self.mag1Label = "mag1"
        self.mag2Label = "mag2"
        self.xRange = None
        self.yRange = None
        
    @classmethod
    def load_arrays(cls, x, y, mag1, mag2, weight=None, header=None):
        """Load a resolved stellar data set by passing position and
        magnitude information as equal-length 1D numpy vectors.
        
        Arguments
        ---------
        x, y : 1D ndarray
           The x and y locations of stars in the field. These should be
           in units of pixels if the WCS is used.
        mag1, mag2 : 1D ndarray
           Magnitude data. `mag1` is the x-axis of the colour-magnitude diagram
           while `mag2` is the y-axis. The axes mag1 and mag2 define the colour
           magnitude (or even colour-colour) space for making stellar
           population selections.
        weight : 1D ndarray (optional)
           Specifies the weight a given point should have relative to others.
           This might be used to correct for completeness from artificial
           star testing. e.g., weight = 1/completeness
        header : a pyfits.Header instance (optional)
           A header instance, in the same frame as the x & y pixel coordinates.
           This is used by the `save_fits()` method to embed a proper WCS
           in the field rendering.
        """
        instance = cls()
        instance.x = x
        instance.y = y
        instance.mag1 = mag1
        instance.mag2 = mag2
        instance.header = header
        instance.nStars = len(x)
        if weight is None:
            instance.weight = np.ones(len(x), dtype=np.float)
        else:
            instance.weight = weight
        # Set bounds of the image area
        if header is not None:
            try:
                instance.xRange = (1, header['NAXIS2'] + 1)
                instance.yRange = (1, header['NAXIS1'] + 1)
            except:
                instance.xRange = (int(np.min(x)), int(np.max(x)))
                instance.yRange = (int(np.min(y)), int(np.max(y)))
        else:
            instance.xRange = (int(np.min(x)), int(np.max(x)))
            instance.yRange = (int(np.min(y)), int(np.max(y)))
        print instance.xRange, instance.yRange
        return instance

    def select_colours(self, poly):
        """Make a stellar population selection in the (mag1,mag2)
        colour-magnitude space using a polygon selection.
        
        Stars are selected using a point-in-polygon code. Selected stars
        are given a value of True in the `selection` member variable.

        Arguments
        ---------
        poly : list
            Each list item is a tuple (mag1_vert,mag2_vert), giving a vertex
            in the colour selection polygon. The polygon closes itself.
        """
        self.selectionPoly = poly
        polyArray = np.array(poly)
        xyPoints = np.array(zip(self.mag1, self.mag2))
        self.selection = mpl.nxutils.points_inside_poly(xyPoints, polyArray)

    def plot_colour_selection(self, plotPath, xLabel=None, yLabel=None):
        """Plot stars in the colour space, along with the selection polygon.
        
        .. todo :: allow for customization of the plot here.
        """
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        inside = np.where(self.selection == True)[0]
        outside = np.where(self.selection == False)[0]
        ax.scatter(self.mag1[inside], self.mag2[inside], s=0.5, c='r',
                marker='o')
        ax.scatter(self.mag1[outside], self.mag2[outside], s=0.5, c='k',
                marker='o')
        polyPatch = mpl.patches.Polygon(self.selectionPoly, closed=True,
                ec='k', fc='r', alpha=0.5, zorder=10)
        ax.add_patch(polyPatch)

        if xLabel is not None: self.mag1Label = xLabel
        if yLabel is not None: self.mag2Label = yLabel
        ax.set_xlabel(self.mag1Label)
        ax.set_ylabel(self.mag2Label)

        fig.savefig(plotPath + ".png", format="png", dpi=300)

    def estimate_density_field(self, targetMass):
        """Runs the density estimation pipeline.
        
        The steps are, in order:

        1. Generate a set of seed nodes
        2. Centroidal voronoi tessellation -- tweak nodes so that
           Voronoi cells encompass roughly equal number of stars
        3. Build a Delaunay tessellation
        4. Estimate density with the Delaunay Tessellation Field Estimator
        5. Render the density field on the Delaunay triangles (interpolation)

        Arguments
        ---------
        targetMass : scalar
            The number of stars (or equivalent, with completeness correction)
            targeted for each Voronoi bin
        """
        # Generate a set of nodes to seed the CVT
        self.generator = ghostmap.EqualMassGenerator()
        self.generator.generate_nodes(self.x, self.y, self.weight, targetMass)
        # Centroidal Voronoi Tessellation -- finds partition so that
        # each cells has approximately equal mass
        self.cvt = CVTessellation(self.x, self.y, self.weight,
                preGenerator=self.generator)
        nodeX, nodeY = self.cvt.get_nodes()
        nodeWeight = self.cvt.get_node_weights()
        # Build a Delaunay tessellation using the CVT nodes
        self.tessellation = ghostmap.DelaunayTessellation(nodeX, nodeY)
        # DTFE Density Estimator -- produces density in stars per pix^2
        dtfe = ghostmap.DelaunayDensityEstimator(self.tessellation)
        nodeDensity = dtfe.estimate_density(self.xRange, self.yRange,
                nodeWeight)
        # Render the density by interpolating over the tessellation
        renderman = ghostmap.FieldRenderer(self.tessellation)
        fieldDensity = renderman.render_first_order_delaunay(nodeDensity,
                self.xRange, self.yRange, 1, 1)
        print "fieldDensity shape", fieldDensity.shape
        self.fieldDensity = fieldDensity

    def save_fits(self, fitsPath):
        """Save the density field as a FITS image to `fitsPath`."""
        if self.header is None:
            pyfits.writeto(fitsPath, self.fieldDensity, clobber=True)
        else:
            pyfits.writeto(fitsPath, self.fieldDensity, self.header,
                           clobber=True)

    def plot_voronoi(self, plotPath, colors=[(0., 0., 0., 1.)]):
        """Diagnostic plot of CMD-selected stars, Voronoi nodes, and
        Voronoi tessellation in the image space.
        """
        fig = plt.figure(figsize=(6, 6))
        fig.subplots_adjust(left=0.15, bottom=0.13, wspace=0.25, right=0.95)
        ax = fig.add_subplot(111, aspect='equal')

        # Plot the original (selected) data
        inside = np.where(self.selection == True)[0]
        ax.scatter(self.x[inside], self.y[inside], s=1., c='r', alpha=0.5,
                marker='o', edgecolor='none')
        
        # Plot the tessellation edges
        tri = self.tessellation.get_triangulation()
        lines = [(tri.circumcenters[i], tri.circumcenters[j])
                    for i in xrange(len(tri.circumcenters))
                        for j in tri.triangle_neighbors[i] if j != -1]
        lines = np.array(lines)
        lc = mpl.collections.LineCollection(lines, colors=colors)
        ax.add_collection(lc)

        # Plot the nodes
        ax.plot(tri.x, tri.y, '.k')
        ax.set_xlim(-50, 550)
        ax.set_ylim(-50, 550)

        fig.savefig(plotPath + ".png", format="png", dpi=300)
        fig.savefig(plotPath + ".pdf", format="pdf")
