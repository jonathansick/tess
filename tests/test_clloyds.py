#!/usr/bin/env python
# encoding: utf-8
"""
Test the _lloyd.c extension module. Simply compute CVT of 100 random
points onto 10 nodes.

2012-10-05 - Created by Jonathan Sick
"""

import numpy as np
from ctypes import c_double, c_long, POINTER
import matplotlib as mpl
import matplotlib.pyplot as plt

import ghostmap.cvtessellation as cvt


def test():
    n = 100
    nNode = 10
    x = np.random.normal(0., 1., n).astype('float64')
    x_ptr = x.ctypes.data_as(POINTER(c_double))
    y = np.random.normal(0., 1., n).astype('float64')
    y_ptr = y.ctypes.data_as(POINTER(c_double))
    w = np.ones(n).astype('float64')
    w_ptr = w.ctypes.data_as(POINTER(c_double))
    xNode = np.random.normal(0., 1., nNode).astype('float64')
    xNode_ptr = xNode.ctypes.data_as(POINTER(c_double))
    yNode = np.random.normal(0., 1., nNode).astype('float64')
    yNode_ptr = yNode.ctypes.data_as(POINTER(c_double))
    vBinNum = np.zeros(n).astype(np.int)
    vBinNum_ptr = vBinNum.ctypes.data_as(POINTER(c_long))

    print xNode
    print yNode

    retval = cvt.lloyd(n, x_ptr, y_ptr, w_ptr,
            nNode, xNode_ptr, yNode_ptr, vBinNum_ptr)
    assert retval == 1, "Lloyd's algorithm failed!"

    print xNode
    print yNode

    plot_nodes(x, y, vBinNum, xNode, yNode)


def plot_nodes(x, y, vBinNum, xNode, yNode):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    nNodes = len(xNode)
    for j in xrange(nNodes):
        idx = np.where(vBinNum == j)
        c = mpl.cm.jet(float(j) / float(nNodes))
        ax.scatter(x[idx], y[idx], edgecolor='None', c=c, alpha=0.5)
        ax.scatter([xNode[j]], [yNode[j]], edgecolor='None', c=c, s=60,
                marker="*")
    fig.savefig("lloyd.pdf", format="pdf")


if __name__ == '__main__':
    test()
