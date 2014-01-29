Issues and TODOs
================

* EqualMassGenerator does not respect point mass; each point is given an equal weight.
* DelaunayDensityEstimtor:
   1. Implemented masked area
   2. Check correction of Triangle area on periphery of tessellation
* The Delaunday field renderer can produce odd results if triangles are smaller than a pixel. This could be solved by doing a post-binning step that re-bins extremely compact bins.
* The Generators can be very slow since they're coded as pure python. A treatment of Cython will fix this.
* There is not support for constrained Voronoi tessellations. As such, it may be best to ignore any cells that make up the peripherly of the tessellation.
* Generally it may be worth refactoring the code which still smells like its IDL ancestor...
