#!/usr/bin/env python
# encoding: utf-8
import os
from setuptools import setup
from distutils import ccompiler

try:
    from sphinx.setup_command import BuildDoc
    cmdclass = {'docs': BuildDoc}
except ImportError:
    BuildDoc = None
    cmdclass = {}


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


# If necessary, use include_dirs=[np.get_include()]) in Extension()
# from https://gist.github.com/3313315
def build_ctypes_ext(name, sources):
    compiler = ccompiler.new_compiler()
    compiler.compile(sources)

    def get_onames(sources):
        o_names = []
        for source in sources:
            head, tail = os.path.split(source)
            o_name = os.path.splitext(tail)[0] + ".o"
            o_names.append(os.path.join(head, o_name))
        return o_names

    compiler.link_shared_object(get_onames(sources), name + ".so")

# Build the ctypes extensions into .so shared libraries
# Included in installation via the package_data field.
# TODO Perhaps exhange the build_ctypes_ext for an autotools-based make setup
# which may be more robust
build_ctypes_ext("ghostmap/_lloyd", ["ghostmap/_lloyd.c"])


setup(
    name="ghostmap",
    packages=["ghostmap"],
    package_data={'': ['*.so']},
    cmdclass=cmdclass,
    author="Jonathan Sick",
    author_email="jonathansick@mac.com",
    description="Delaunay and Voronoi tessellation for astronomers.",
    long_description=read("README.md"),
    url="https://github.com/jonathansick/ghostmap"
)
