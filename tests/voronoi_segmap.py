#!/usr/bin/env python
# encoding: utf-8
"""
Demo of Voronoi pixel segmentation maps and assigning points to Voronoi
bins.

2012-10-26 - Created by Jonathan Sick
"""

import numpy as np
import astropy.io.fits

from tess import ghostmap
from tess.cvtessellation import CVTessellation


def main():
    # Set up the size of the mock image
    xRange = [0, 500]
    yRange = [0, 500]
    # We'll generate data distributed in a 2D guassian space
    x, y = guassian_point_process(250, 250, 100, 50, 20000)
    inRange = np.where((x > 0) & (y > 0)
            & (x < xRange[1]) & (y < yRange[1]))[0]
    x = x[inRange]
    y = y[inRange]
    nStars = len(x)
    weight = np.random.uniform(0.1, 1., size=(nStars,))

    # Centroidal Voronoi Tessellation, meeting a target cell mass
    targetMass = 20
    gen = ghostmap.EqualMassGenerator()
    gen.generate_nodes(x, y, weight, targetMass)
    cvt = CVTessellation(x, y, weight, preGenerator=gen)

    # Set up the pixel grid (or use set_fits_grid if using a FITS image)
    cvt.set_pixel_grid(xRange, yRange)

    # Map of Voronoi cell IDs (saving to FITS)
    cvt.save_segmap("voronoi_segmap.fits")

    # Compute Density of Points and save to FITS
    cvt.compute_cell_areas()
    cellDensity = cvt.cell_point_density(x, y, mass=weight)
    densityField = cvt.render_voronoi_field(cellDensity)
    astropy.io.fits.writeto("voronoi_densitymap.fits", densityField, clobber=True)

    # Save CSV table of Voronoi bins, areas, and densities
    save_cell_table(cvt, cellDensity)

    # Save a list of points with assignment to Voronoi cells
    save_cell_membership(cvt, x, y)


def save_cell_table(cvt, cellDensity):
    """Save a table of Voronoi cells. Fields are
    
    1. Voronoi Cell ID
    2. Node X (coordinates of cell centers)
    3. Node Y
    4. Cell Area (pixels squared)
    5. Cell Density (density of points in each Voronoi cell
    """
    cellID = np.arange(len(cvt.xNode), dtype=int)
    data = np.asarray((cellID, cvt.xNode, cvt.yNode, cvt.cellAreas,
        cellDensity)).T
    np.savetxt("voronoi_cells.txt", data,
            fmt=("%i", "%.1f", "%.1f", "%.3f", "%.6f"))


def save_cell_membership(cvt, x, y):
    """Save a table assigning each point (x, y) to a Voronoi cell.
    
    Format:

    1. Point X
    2. Point Y
    3. Voronoi Cell ID
    """
    cellIDs = cvt.partition_points(x, y)
    data = np.asarray((x, y, cellIDs)).T
    data = np.savetxt("voronoi_point_partition.txt",
            data, fmt=("%.1f", "%.1f", "%i"))


def guassian_point_process(x0, y0, xSigma, ySigma, nPoints):
    """Returns a x and y coordinates of points sampled from a
    2D guassian dist."""
    x = np.random.normal(loc=x0, scale=xSigma, size=(nPoints,))
    y = np.random.normal(loc=y0, scale=ySigma, size=(nPoints,))
    return x, y


if __name__ == '__main__':
    main()
