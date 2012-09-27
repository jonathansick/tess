#!/usr/bin/env python
# encoding: utf-8
"""
Python ctypes wrapper for _lloyd.c

2012-09-27 - Created by Jonathan Sick
"""

import ctypes
import numpy as np

from ctypes import c_double, c_long, pointer, POINTER


def _load_function(dllPath, fcnName, fcnArgTypes=None):
    """Load the .so file with ctypes.
    
    This function is largely lifted from
    https://gist.github.com/3313315
    """
    dll = ctypes.CDLL(dllPath, mode=ctypes.RTLD_GLOBAL)

    # Get reference to function symbol in DLL
    print dll.__dict__
    # Note we use exec to dynamically write the code based on the
    # user's `fcnName` input and save to variable func
    func = None
    exec "func = " + ".".join(("dll", fcnName))
    print "loaded function", func

    # Set call signature for safety
    if fcnArgTypes is not None:
        func.argtypes = fcnArgTypes

    return func

# Load Lloyd function
lloyd = _load_function("./_lloyd.so", "lloyd",
        [POINTER(c_double), POINTER(c_double), POINTER(c_double),
         POINTER(c_double), POINTER(c_double), POINTER(c_long)])


def test():
    n = 10
    nNode = 2
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

    print lloyd(x_ptr, y_ptr, w_ptr, xNode_ptr, yNode_ptr, vBinNum_ptr)


if __name__ == '__main__':
    test()
