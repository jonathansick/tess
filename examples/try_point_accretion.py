#!/usr/bin/env python
# encoding: utf-8
"""
Burn-in test for the point accretion code
"""

import numpy as np
from tess.point_accretion import EqualMassAccretor


def main():
    n_points = 100
    mass = np.random.rand(n_points)
    xy = np.random.randn(n_points, 2)
    accretor = EqualMassAccretor(xy, mass, 5.)
    accretor.accrete()


if __name__ == '__main__':
    main()
