from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

"""
matrices
==========

Often used matrices
"""

import scipy


def zeros(dim):
    return scipy.zeros((dim,dim))

def identity(dim):
    return scipy.identity(dim)
#
# Pauli spin 1/2 operators:
#
def sigma0():
    return scipy.array([[0., 0.], [0., 0.]])
    
def sigmai():
    return scipy.array([[1., 0.], [0., 1.]])

def sigmap():
    return scipy.array([[0., 1.], [0., 0.]])

def sigmam():
    return scipy.array([[0., 0.], [1., 0.]])

def sigmax():
    return scipy.array([[0., 1.], [1., 0.]])

def sigmay():
    return  scipy.array([[0., -1.j], [1.j, 0.]])

def sigmaz():
    return scipy.array([[1., 0.], [0., -1.]])


def annihilation(dim):
    """
    Destruction (lowering) operator.

    Parameters
    ----------
    dim : int
        Dimension of Hilbert space.
    """

    return scipy.diag(scipy.sqrt(scipy.arange(1,dim)), 1)


def creation(dim):
    """
    Creation (raising) operator.

    Parameters
    ----------
    dim : int
        Dimension of Hilbert space.
    """
    return scipy.diag(scipy.sqrt(scipy.arange(1,dim)), -1)


def basis(dim, i):
    """
    dim x 1 column vector with all zeros except a one at row i
    """
    vec = scipy.zeros(dim)
    vec[i] = 1.0
    return scipy.array(vec)