Overview of Binning with Centroidal Voronoi Tessellations
=========================================================

`ghostmap` is a Python module that allows you to bin a two-dimensional distribution of points into discrete Voronoi cells. Voronoi binning has the great property that cells--the 2D 'bins'--are sized and located according to the underlying point distribution. Where there is a high density (or a high *S/N*), the cells are smaller; where point distribution is sparse the cells are larger. In fact, `ghostmap` uses Lloyd's method fo Centroidal Voronoi Tessellation (CVT) so that *every cell has equal mass, or equal S/N.*

Code Architecture
-----------------

`ghostmap` is intended to be used at the code-level. The user writes a Python script/pipeline that loads the data, executes the binning, and extracts the binning result. Lets run through the basic `ghostmap` classes that the user will use.

Point Data Storage
   `PointList2D`.

Generators
   `AccretionGenerator`: `EqualSNGenerator` or `EqualMassGenerator`.

Centroidal Voronoi Tessellation
   `CVTessellation`

Delaunay Triangulation
   `Delaunay Triangulation`

Density Estimators
   `Voronoi Density Estimator` or `Delaunay Density Estimator`

Field Rendering
   Given the Delaunay Triangulation and the node values (*e.g.*, the point density or some other statistic for a bin), the use can render the field. Using the `FieldRenderer` class, 2D numpy arrays (images) can be made. There are several methods for rendering the field:

   * `renderZerothOrderVoronoi()` will simply create an image where each pixel in a cell is given the value of node. This creates a tiled look.
   * `renderFirstOrderDelaunay()` will use a Delaunay triangulation that interpolates the values of the nodes (which are the vertices of the triangulation). This can be through of as a linear interpolation.
   * `renderNearestNeighboursDelaunay()` will use a nearest-neighbours interpolation to create a small rendering of the field. This is not gauranteed to conserve mass in a density rendering.
