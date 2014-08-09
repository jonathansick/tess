User's Guide to Tess
====================

Tess is a Python package that helps astronomers use Voronoi and Delaunay tessellations to work with point and pixel spatial datasets.
Supported tasks include pixel and point binning, estimating density fields, and rendering fields with Voronoi and Delaunay geometries.
The tutorials in this documentation should give a good idea of the pipelines that can be built with Tess.


Installation
------------

Tess is not yet on PyPI, but can still be installed from GitHub::

   pip install git+git://github.com/jonathansick/tess.git

Alternatively, you're welcome to clone `Tess from GitHub <http://github.com/jonathansick/tess>`_ and install using the usual ``python setup.py install`` command.

Tess depends on Astropy, Numpy, Scipy, Matplotlib and Cython.


Contents
--------

.. toctree::
   :maxdepth: 2

   voronoi_tutorial
   density_tutorial
   api/index


Inspiration
-----------

Tess grew directly from the inspiration of `Cappellari & Copin (2003)`_, which introduced the Centroidal Voronoi Tessellation to astronomy.
The methods used for estimating 2D density fields with Delaunay tessellations (*i.e.*, the DTFE) are based on `Schaap &  van de Weygaert (2000)`_.


.. _Cappellari & Copin (2003): http://adsabs.harvard.edu/abs/2003MNRAS.342..345C
.. _Schaap &  van de Weygaert (2000): http://adsabs.harvard.edu/abs/2000A%26A...363L..29S


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
