Overview of Binning with Centroidal Voronoi Tessellations
=========================================================

`ghostmap` is a Python module that allows you to bin a two-dimensional distribution of points into discrete Voronoi cells.
Voronoi binning has the great property that cells--the 2D 'bins'--are sized and located according to the underlying point distribution.
Where there is a high density (or a high *S/N*), the cells are smaller; where point distribution is sparse the cells are larger.
In fact, `ghostmap` uses Lloyd's method of Centroidal Voronoi Tessellation (CVT) so that *every cell has equal mass, or equal S/N.*
This 

Architecture
------------

`ghostmap` is intended to be used as a package called by your own Python analysis pipeline.
Lets run through the basic `ghostmap` classes that the user will use.

Generators
   The first step in spatial binning is to choose a set of nodes that *generate* the tessellation.
   These generators aren't precise; the Lloyd's method of Centroidal Voronoi Tessellation will tweak the positions to ensure that *S/N* is maintained across Voronoi cells.
   
   `ghostmap` supports two choices for generators: equal S/N (:class:`ghostmap.EqualSNGenerator`), or equal mass (:class:`ghostmap.EqualMassGenerator`).

Centroidal Voronoi Tessellation
   :class:`CVTessellation`

Delaunay Triangulation
   `Delaunay Triangulation`

Density Estimators
   `Voronoi Density Estimator` or `Delaunay Density Estimator`

Field Rendering
   Given the Delaunay Triangulation and the node values (*e.g.*, the point density or some other statistic for a bin), the use can render the field. Using the `FieldRenderer` class, 2D numpy arrays (images) can be made. There are several methods for rendering the field:

   * `renderZerothOrderVoronoi()` will simply create an image where each pixel in a cell is given the value of node. This creates a tiled look.
   * `renderFirstOrderDelaunay()` will use a Delaunay triangulation that interpolates the values of the nodes (which are the vertices of the triangulation). This can be through of as a linear interpolation.
   * `renderNearestNeighboursDelaunay()` will use a nearest-neighbours interpolation to create a small rendering of the field. This is not gauranteed to conserve mass in a density rendering.
