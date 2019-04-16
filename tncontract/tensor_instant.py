from __future__ import (absolute_import, division,
						print_function, unicode_literals)

__all__ = ['random_tensor', "zeros_tensor", "identity_tensor", "zeros_tensor_like"]

import copy

from tncontract.tnxp import xp as xp
from tncontract import tensor_core as tnc


def random_tensor(shape, labels=None, base_label="i", dtype=float):
	if dtype==float:
		data = xp.random.rand(*shape)*2-1
	elif dtype==complex:
		data = (xp.random.rand(*shape)*2-1) + 1j*(xp.random.rand(*shape)*2-1)
	#elif dtype==int:
	#	data = xp.random.randint(0,10,size=shape)
	else:
		raise ValueError("random_tensor argument dtype must be float or complex")
	return tnc.Tensor(data, labels=labels, base_label=base_label)


def zeros_tensor(shape, labels=None, base_label="i", dtype=float):
	data = xp.zeros(shape, dtype=dtype)
	return tnc.Tensor(data, labels=labels, base_label=base_label)


def identity_tensor(physDim, physoutLabel="physout_byIdentity", physinLabel="physin_byIdentity", virtLabels=None, dtype=float):
	matrix = xp.identity(physDim, dtype=dtype)
	tensor = tnc.Tensor(matrix, labels=[physoutLabel, physinLabel])
	if not virtLabels is None:
		for label in virtLabels:
			tensor.add_dummy_index(label)
	return tensor


def random_tensor_like(tensor):
	return random_tensor(tensor.shape, labels=copy.copy(tensor.labels), dtype=tensor.dtype)


def zeros_tensor_like(tensor):
	return zeros_tensor(tensor.shape, labels=copy.copy(tensor.labels), dtype=tensor.dtype)