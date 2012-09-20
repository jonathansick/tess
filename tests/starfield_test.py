#!/usr/bin/env python
# encoding: utf-8
"""
Mock data test script for StarField class

2012-09-20 - Created by Jonathan Sick
"""

import numpy as np
from ghostmap.starfield import StarField


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
    # Generate a distribution of 'colours'
    mag2 = np.random.uniform(0., 10., size=(nStars,))
    mag1 = np.random.normal(loc=5., scale=2., size=(nStars,))
    print mag1.shape, mag2.shape
    
    # Use StarField as an interface to the tessellation and
    # density estimation pipeline
    starField = StarField.load_arrays(x, y, mag1, mag2, weight=weight)
    starField.select_colours([(2., 1.), (2., 8.), (5., 6.), (4., 1.)])
    starField.plot_colour_selection("test_cmd_selection",
            xLabel="mag1", yLabel="mag2")
    # 20 is the target number of stars in each cell
    starField.estimate_density_field(20.)
    starField.save_fits("test_density_field.fits")
    starField.plot_voronoi("test_voronoi_diagram")


def guassian_point_process(x0, y0, xSigma, ySigma, nPoints):
    """Returns a x and y coordinates of points sampled from a
    2D guassian dist."""
    x = np.random.normal(loc=x0, scale=xSigma, size=(nPoints,))
    y = np.random.normal(loc=y0, scale=ySigma, size=(nPoints,))
    return x, y


if __name__ == '__main__':
    main()
