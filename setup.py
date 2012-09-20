#!/usr/bin/env python
# encoding: utf-8
import os
from setuptools import setup

try:
    from sphinx.setup_command import BuildDoc
    cmdclass = {'docs': BuildDoc}
except ImportError:
    BuildDoc = None
    cmdclass = {}


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="ghostmap",
    packages=["ghostmap"],
    cmdclass=cmdclass,
    author="Jonathan Sick",
    author_email="jonathansick@mac.com",
    description="Delaunay and Voronoi tessellation for astronomers.",
    long_description=read("README.md"),
    url="https://github.com/jonathansick/ghostmap"
)
