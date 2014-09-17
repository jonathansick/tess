====
Tess
====

Tess is a Python package for helping astronomers use Voronoi and Delaunay tessellations to study and organize spatial data sets.
Tess is meant to work with both sparse spatial datasets (points) and pixel data sets.
We make use of Cython wherever possible for speed and portability.

Tess originated from the GHOSTS survey (PI: Roelof de Jong), and is now being developed for the ANDROIDS project (PI: Jonathan Sick).
In turn it was inspired by the `IDL code <http://www-astro.physics.ox.ac.uk/~mxc/software/>`_  of `Cappellari and Copin (2003) <http://adsabs.harvard.edu/abs/2003MNRAS.342..345C>`_
Please let me know if this package is useful for you, and let me know how you're using it.

The `documentation has some examples that might answer your questions. <http://http://jonathansick.github.io/tess/tess/index.html>`_


------------
Installation
------------

Pretty straightforward::

    git clone git@github.com:jonathansick/tess.git
    cd tess
    python setup.py install


Tess requires numpy, cython and matplotlib.


----
Info
----

Copyright 2014 Jonathan Sick, @jonathansick

BSD Licensed.
