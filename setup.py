#!/usr/bin/env python
# encoding: utf-8
import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="ghostmap",
    packages=["ghostmap"],
    author="Jonathan Sick",
    author_email="jonathansick@mac.com",
    description="Delaunay and Voronoi tessellation for astronomers.",
    long_description=read("README.md"),
    url="https://github.com/jonathansick/ghostmap"
)
