.. ghostmap documentation master file, created by
   sphinx-quickstart on Thu Jul 14 19:08:01 2011.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Ghostmap --- Voronoi binning and density fields for astronomy
=============================================================

Ghostmap is a Python package that supports three basic, and related, analysis tasks:

1. Adaptively binning a spatial dataset to maximize resolution and S/N.
2. Estimating the density field in a spatial dataset
3. Rendering a density field, or any other field, using the tessellation.


Contents
--------

.. toctree::
   :maxdepth: 2
   
   vt_basics
   todo
   voronoitessellation
   cvtessellation

Inspiration
-----------

Ghostmap grew directly from the inspiration of `Cappellari & Copin (2003)`_, which introduced the Centroidal Voronoi Tessellation to astronomy.
The methods used for estimating 2D density fields with Delaunay tessellations (*i.e.*, the DTFE) are based on `Schaap &  van de Weygaert (2000)`_.


.. _Cappellari & Copin (2003): http://adsabs.harvard.edu/abs/2003MNRAS.342..345C
.. _Schaap &  van de Weygaert (2000): http://adsabs.harvard.edu/abs/2000A%26A...363L..29S

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
