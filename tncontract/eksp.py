#cupy tukau tame ni toriaezu tukatteruyatu zenbu kaite mitemasu
"""
first arg is numpy.ndarray:
	numpy:
		asarray, array, rollaxis, moveaxis, reshape, pad, trace, conj, tensordot, diag, sqrt, concatenate, dot
	scipy.linalg:
		norm, svd, qr, eigh
	scipy.sparse.linalg:
		eigsh

not:
	numpy:
		newaxis, zeros, identity, arange
	numpy.linalg:
		LinAlgError, 
	numpy.random:
		rand, 
	scipy.sparse.linalg:
		LinearOperator

"""




class Xp():
	def __init__(self):
		self.use_scipy()

	def use_scipy(self):
		self.isScipy = True
		self.isCupy = False
		import scipy
		import scipy.linalg
		import scipy.sparse.linalg
		import numpy.random
		self._core = scipy
		self.linalg = scipy.linalg
		self.sparse.linalg = scipy.sparse.linalg
		self.random = numpy.random

	def use_cupy(self):
		self.isCupy = True
		self.isScipy = False
		import cupy
		import cupy.linalg
		import cupy.random
		self._core = cupy
		self.linalg = cupy.linalg
		self.random = cupy.random

	def __getattr__(self, name):
		return self._core.__getattribute__(name)

xp = Xp()
