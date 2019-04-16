from __future__ import (absolute_import, division,
						print_function, unicode_literals)

__all__ = ['Tensor', 'contract', 'tensor_product', "directSumTensor", 'matrix_to_tensor',
		   'tensor_to_matrix', "vector_to_tensor", "tensor_to_vector", 'tensor_svd',
		   'truncated_svd']

import copy as copyModule
import warnings
import numpy
from tncontract.tnxp import xp as xp
from tncontract import label as lbl


class Tensor():
	def __init__(self, data, labels=None, base_label="i", copy=False):
		if copy:
			self.data = xp.asarray(data)
		else:
			if isinstance(data, xp.ndarray):
				self.data = data
			else:
				self.data = xp.asarray(data)
		if labels is None:
			self.assign_labels(base_label=base_label)
		else:
			if copy:
				self.labels = copyModule.copy(labels)
			else:
				self.labels = labels

	def copy(self):
		return Tensor(data=self.data, labels=self.labels, copy=True)

	#@property
	#def xp(self):
	#	return cupy.get_array_module(self.data)


	#methods for repr, str
	def __repr__(self):
		return "Tensor(data=%r, labels=%r)" % (self.data, self.labels)

	def __str__(self):
		array_str = str(self.data)
		lines = array_str.splitlines()
		if len(lines) > 20:
			lines = lines[:20] + ["...",
								  "Printed output of large array was truncated.\nString "
								  "representation of full data array returned by "
								  "tensor.data.__str__()."]
			array_str = "\n".join(lines)

		# Specify how index information is printed
		lines = []
		for i, label in enumerate(self.labels):
			lines.append("   " + str(i) + ". (dim=" + str(self.shape[i]) + ") " +
						 str(label) + "\n")
		indices_str = "".join(lines)

		return ("Tensor object: \n" +
				"Data type: " + str(self.data.dtype) + "\n"
						"Number of indices: " + str(len(self.data.shape)) + "\n"
								"\nIndex labels:\n" + indices_str +
				# "shape = " + str(self.shape) +
				# ", labels = " + str(self.labels) + "\n" +(
				"\nTensor data = \n" + array_str)
	#properties
	@property
	def shape(self):
		return self.data.shape

	@property
	def rank(self):
		return len(self.shape)
		
	@property
	def dtype(self):
		return self.data.dtype

	def index_of_label(self, label):
		return self.labels.index(label)

	def dimension_of_label(self, label):
		index = self.index_of_label(label)
		return self.data.shape[index]

	index_dimension = dimension_of_label
	dimension_of_index = dimension_of_label


	#methods for ToContract
	def __getitem__(self, *args):
		return ToContract(self, *args)


	#methods for labels
	def get_labels(self):
		return self._labels

	def set_labels(self, labels):
		if len(labels) != len(self.data.shape):
			raise ValueError("Labels do not match shape of data. labels=={0}, shape=={1}".format(labels, self.data.shape))
		if len(labels) != len(set(labels)):
			raise ValueError("Labels are not unique. labels=={0}".format(labels))
		self._labels = list(labels)

	labels = property(get_labels, set_labels)

	def assign_labels(self, base_label="i"):
		self.labels = [base_label + str(i) for i in range(len(self.data.shape))]

	def replace_label(self, old_labels, new_labels):
		if not isinstance(old_labels, list):
			old_labels = [old_labels]
		if not isinstance(new_labels, list):
			new_labels = [new_labels]

		for i, label in enumerate(self.labels):
			if label in old_labels:
				self.labels[i] = new_labels[old_labels.index(label)]


	#methods for move
	def move_index_to_top(self, labelMove):
		indexMoveFrom = self.index_of_label(labelMove)
		self.labels.pop(indexMoveFrom)
		self.labels.insert(0, labelMove)
		self.data = xp.rollaxis(self.data, indexMoveFrom, 0)

	def move_index_to_bottom(self, labelMove):
		indexMoveFrom = self.index_of_label(labelMove)
		self.labels.pop(indexMoveFrom)
		self.labels.append(labelMove)
		self.data = xp.rollaxis(self.data, indexMoveFrom, self.rank)

	def move_index_to_position(self, labelMove, position):
		indexMoveFrom = self.index_of_label(labelMove)
		if position==indexMoveFrom:
			pass
		elif position<indexMoveFrom:
			self.labels.pop(indexMoveFrom)
			self.labels.insert(position, labelMove)
			self.data = xp.rollaxis(self.data, indexMoveFrom, position)
		else:
			self.labels.pop(indexMoveFrom)
			self.labels.insert(position, labelMove)
			self.data = xp.rollaxis(self.data, indexMoveFrom, position+1)

	def move_indices_to_top(self, labelsMove):
		if isinstance(labelsMove, tuple):
			labelsMove = list(labelsMove)
		elif not isinstance(labelsMove, list):
			labelsMove = [labelsMove]

		oldIndicesMoveFrom = [self.index_of_label(label) for label in labelsMove]
		newIndicesMoveTo = list(range(len(oldIndicesMoveFrom)))

		oldIndicesNotMoveFrom = [i for i in range(len(self.labels)) if not i in oldIndicesMoveFrom]
		#newIndicesNotMoveTo = list(range(len(oldIndicesMoveFrom), len(self.labels)))

		oldLabels = self.labels
		newLabels = []
		for oldIndex in oldIndicesMoveFrom:
			newLabels.append(oldLabels[oldIndex])
		for oldIndex in oldIndicesNotMoveFrom:
			newLabels.append(oldLabels[oldIndex])

		self.data = xp.moveaxis(self.data, oldIndicesMoveFrom, newIndicesMoveTo)
		self.labels = newLabels

	def move_indices_to_bottom(self, labelsMove):
		if not isinstance(labelsMove, list):
			labelsMove = [labelsMove]

		oldIndicesMoveFrom = [self.index_of_label(label) for label in labelsMove]
		newIndicesMoveTo = list(range(self.rank-len(oldIndicesMoveFrom), self.rank))

		oldIndicesNotMoveFrom = [i for i in range(len(self.labels)) if not i in oldIndicesMoveFrom]
		#newIndicesNotMoveTo = list(range(self.rank-len(oldIndicesMoveFrom)))

		oldLabels = self.labels
		newLabels = []
		for oldIndex in oldIndicesNotMoveFrom:
			newLabels.append(oldLabels[oldIndex])
		for oldIndex in oldIndicesMoveFrom:
			newLabels.append(oldLabels[oldIndex])

		self.data = xp.moveaxis(self.data, oldIndicesMoveFrom, newIndicesMoveTo)
		self.labels = newLabels

	def move_indices_to_position(self, labelsMove, position):
		if not isinstance(labelsMove, list):
			labelsMove = [labelsMove]

		oldIndicesMoveFrom = [self.index_of_label(label) for label in labelsMove]
		newIndicesMoveTo = list(range(position, position+len(labelsMove)))

		oldIndicesNotMoveFrom = [i for i in range(len(self.labels)) if not i in oldIndicesMoveFrom]
		newIndicesNotMoveTo = list(range(position)) + list(range(position+len(labelsMove), self.rank))

		oldLabels = self.labels
		newLabels = [None]*len(oldLabels)
		for oldIndex, newIndex in zip(oldIndicesMoveFrom, newIndicesMoveTo):
			newLabels[newIndex] = oldLabels[oldIndex]
		for oldIndex, newIndex in zip(oldIndicesNotMoveFrom, newIndicesNotMoveTo):
			newLabels[newIndex] = oldLabels[oldIndex]

		self.data = xp.moveaxis(self.data, oldIndicesMoveFrom, newIndicesMoveTo)
		self.labels = newLabels


	#methods for fuse, split
	def fuse_indices(self, labelsFuse, newLabel):
		self.move_indices_to_top(labelsFuse)

		oldShape = self.data.shape
		oldShapeFuse = oldShape[:len(labelsFuse)]
		shapeNotFuse = oldShape[len(labelsFuse):]
		newDimFuse = numpy.prod(oldShapeFuse, dtype=int)
		newShape = (newDimFuse,) + shapeNotFuse

		oldLabels = self.labels
		newLabels = [newLabel]+oldLabels[len(labelsFuse):]

		self.data = xp.reshape(self.data, newShape)
		self.labels = newLabels

		return oldShapeFuse, newDimFuse

	def split_index(self, labelSplit, newShapeSplit, newLabelsSplit):
		if len(newShapeSplit) != len(newLabelsSplit):
			raise ValueError("Length of new_dims must equal length of "
							 "new_labels")

		newShapeSplit = tuple(newShapeSplit)
		indexSplit = self.index_of_label(labelSplit)
		newShape = self.data.shape[:indexSplit] + newShapeSplit + self.data.shape[indexSplit + 1:]
		newLabels = self.labels[:indexSplit] + newLabelsSplit + self.labels[indexSplit + 1:]

		self.data = xp.reshape(self.data, newShape)
		self.labels = newLabels



	def conjugate(self):
		self.data = self.data.conj()

	def conjugated(self):
		return Tensor(data=self.data.conj(), labels=copyModule.copy(self.labels))

	#methods for mul, add
	def __imul__(self, scalar):
		try:
			self.data *= scalar
			return self
		except:
			return NotImplemented

	def __mul__(self, scalar):
		try:
			out = Tensor(self.data*scalar, labels=copyModule.copy(self.labels))
			return out
		except:
			return NotImplemented

	def __rmul__(self, scalar):
		try:
			out = Tensor(self.data*scalar, labels=copyModule.copy(self.labels))
			return out
		except:
			return NotImplemented

	def __itruediv__(self, scalar):
		try:
			self.data /= scalar
			return self
		except:
			return NotImplemented

	def __truediv__(self, scalar):
		try:
			out = Tensor(self.data/scalar, labels=copyModule.copy(self.labels))
			return out
		except:
			return NotImplemented

	def __iadd__(self, other, skipLabelSort=False):
		try:
			if not skipLabelSort:
				self.move_indices_to_top(other.labels)
			self.data += other.data
			return self
		except:
			return NotImplemented

	def __add__(self, other, skipLabelSort=False):
		try:
			temp = self.copy()
			if not skipLabelSort:
				temp.move_indices_to_top(other.labels)
			temp.data += other.data
			return temp
		except Exception as e:
			raise e
			return NotImplemented

	def __isub__(self, other):
		try:
			self.move_indices_to_top(other.labels)
			self.data -= other.data
			return self
		except:
			return NotImplemented

	def __sub__(self, other):
		copySelf = self.copy()
		copySelf -= other
		return copySelf

	def pad_indices(self, labels, npads):
		wholeNpad = [(0,0)] * self.rank
		for label,npad in zip(labels, npads):
			index = self.index_of_label(label)
			wholeNpad[index] = npad
		self.data = xp.pad(self.data, wholeNpad, mode="constant", constant_values=0)

	def padded_indices(self, labels, npads):
		indices = [self.index_of_label(label) for label in labels]
		wholeNpad = [(0,0)] * self.rank
		for index,npad in zip(labels, npads):
			wholeNpad[index] = npad
		newData = xp.pad(self.data, npad, mode="constant", constant_values=0)
		return Tensor(newData, labels=copyModule.copy(self.labels))

	def norm(self):
		"""Return the frobenius norm of the tensor, equivalent to taking the
		sum of absolute values squared of every element. """
		return xp.linalg.norm(self.data)

	def normalise(self):
		norm = self.norm()
		self.data /= norm



	#methods for trace, contract
	def contracted_internal(self, label1, label2):
		index1 = self.index_of_label(label1)
		index2 = self.index_of_label(label2)
		index1,index2 = min(index1,index2), max(index1,index2)

		newData = xp.trace(self.data, axis1=index1, axis2=index2)
		newLabels = self.labels[:index1]+self.labels[index1+1:index2]+self.labels[index2+1:]

		return Tensor(newData, newLabels)

	def contract_internal(self, label1, label2):
		index1 = self.index_of_label(label1)
		index2 = self.index_of_label(label2)
		index1,index2 = min(index1,index2), max(index1,index2)

		self.data = xp.trace(self.data, axis1=index1, axis2=index2)
		self.labels = self.labels[:index1]+self.labels[index1+1:index2]+self.labels[index2+1:]

	traced = contracted_internal
	trace = contract_internal
	tr = contract_internal

	def contract(self, *args, **kwargs):
		"""
		A method that calls the function `contract`, passing `self` as the
		first argument.
		
		See also
		--------
		contract (function)

		"""
		t = contract(self, *args, **kwargs)
		self.data = t.data
		self.labels = t.labels


	#methods for dummy index
	def add_dummy_index(self, label):
		"""Add an additional index to the tensor with dimension 1, and label 
		specified by the index "label". The position argument specifies where 
		the index will be inserted. """
		# Will insert an axis of length 1 in the first position
		self.data = self.data[xp.newaxis, :]
		self.labels.insert(0, label)

	def remove_all_dummy_indices(self, labels=None):
		"""Removes all dummy indices (i.e. indices with dimension 1)
		which have labels specified by the labels argument. None
		for the labels argument implies all labels."""
		orig_shape = self.shape
		for i, x in enumerate(self.labels):
			if orig_shape[i]==1 and ((labels is None) or (x in labels)):
				self.move_index_to_top(x)
				self.data = self.data[0]
				self.labels = self.labels[1:]


	#methods for convert
	def to_matrix(self, row_labels, column_labels=None):
		return tensor_to_matrix(self, row_labels, column_labels)

	def to_vector(self, row_labels):
		return tensor_to_vector(self, row_labels)


	#ufufu
	def forceHermite(self,physout_label,physin_label):
		physout_index = self.index_of_label(physout_label)
		physin_index = self.index_of_label(physin_label)
		a = self.data
		b = xp.swapaxes(a,physout_index,physin_index)
		b = xp.conj(b)
		c = (a + b)/2
		self.data = c


class ToContract():
	"""A simple class that contains a Tensor and a list of indices (labels) of
	that tensor which are to be contracted with another tensor. Used in
	__mul__, __rmul__ for convenient tensor contraction."""

	def __init__(self, tensor, labels):
		self.tensor = tensor
		self.labels = labels

	def __mul__(self, other):
		# If label argument is not a tuple, simply use that as the argument to
		# contract function. Otherwise convert to a list.
		if not isinstance(self.labels, tuple):
			labels1 = self.labels
		else:
			labels1 = list(self.labels)
		if not isinstance(other.labels, tuple):
			labels2 = other.labels
		else:
			labels2 = list(other.labels)
		return contract(self.tensor, other.tensor, labels1, labels2)

		# Tensor constructors


	


def contract(tensor1, tensor2, labels1, labels2, index_slice1=None,
			 index_slice2=None):
	"""
	Contract the indices of `tensor1` specified in `labels1` with the indices
	of `tensor2` specified in `labels2`. 
	
	This is an intuitive wrapper for numpy's `tensordot` function.  A pairwise
	tensor contraction is specified by a pair of tensors `tensor1` and
	`tensor2`, a set of index labels `labels1` from `tensor1`, and a set of
	index labels `labels2` from `tensor2`. All indices of `tensor1` with label
	in `labels1` are fused (preserving order) into a single label, and likewise
	for `tensor2`, then these two fused indices are contracted. 

	Parameters
	----------
	tensor1, tensor2 : Tensor
		The two tensors to be contracted.

	labels1, labels2 : str or list
		The indices of `tensor1` and `tensor2` to be contracted. Can either be
		a single label, or a list of labels. 

	Examples
	--------
	Define a random 2x2 tensor with index labels "spam" and "eggs" and a random
	2x3x2x4 tensor with index labels 'i0', 'i1', etc. 

	>>> A = random_tensor(2, 2, labels = ["spam", "eggs"])
	>>> B = random_tensor(2, 3, 2, 4)
	>>> print(B)
	Tensor object: shape = (2, 3, 2, 4), labels = ['i0', 'i1', 'i2', 'i3']
	
	Contract the "spam" index of tensor A with the "i2" index of tensor B.
	>>> C = contract(A, B, "spam", "i2")
	>>> print(C)
	Tensor object: shape = (2, 2, 3, 4), labels = ['eggs', 'i0', 'i1', 'i3']

	Contract the "spam" index of tensor A with the "i0" index of tensor B and
	also contract the "eggs" index of tensor A with the "i2" index of tensor B.

	>>> D = contract(A, B, ["spam", "eggs"], ["i0", "i2"])
	>>> print(D)
	Tensor object: shape = (3, 4), labels = ['i1', 'i3']

	Note that the following shorthand can be used to perform the same operation
	described above.
	>>> D = A["spam", "eggs"]*B["i0", "i2"]
	>>> print(D)
	Tensor object: shape = (3, 4), labels = ['i1', 'i3']

	Returns
	-------
	C : Tensor
		The result of the tensor contraction. Regarding the `data` and `labels`
		attributes of this tensor, `C` will have all of the uncontracted
		indices of `tensor1` and `tensor2`, with the indices of `tensor1`
		always coming before those of `tensor2`, and their internal order
		preserved. 

	"""

	# If the input labels is not a list, convert to list with one entry
	if not isinstance(labels1, list):
		labels1 = [labels1]
	if not isinstance(labels2, list):
		labels2 = [labels2]

	tensor1_indices = []
	for label in labels1:
		# Append all indices to tensor1_indices with label
		tensor1_indices.extend([i for i, x in enumerate(tensor1.labels)
								if x == label])

	tensor2_indices = []
	for label in labels2:
		# Append all indices to tensor1_indices with label
		tensor2_indices.extend([i for i, x in enumerate(tensor2.labels)
								if x == label])

	if len(labels1) != len(tensor1_indices) or len(labels2) != len(tensor2_indices):
		raise Exception("labels1="+str(labels1)+",\nlabels2="+str(labels2))

	# Replace the index -1 with the len(tensor1_indeces),
	# to refer to the last element in the list
	if index_slice1 is not None:
		index_slice1 = [x if x != -1 else len(tensor1_indices) - 1 for x
						in index_slice1]
	if index_slice2 is not None:
		index_slice2 = [x if x != -1 else len(tensor2_indices) - 1 for x
						in index_slice2]

	# Select some subset or permutation of these indices if specified
	# If no list is specified, contract all indices with the specified labels
	# If an empty list is specified, no indices will be contracted
	if index_slice1 is not None:
		tensor1_indices = [j for i, j in enumerate(tensor1_indices)
						   if i in index_slice1]
	if index_slice2 is not None:
		tensor2_indices = [j for i, j in enumerate(tensor2_indices)
						   if i in index_slice2]

	# Contract the two tensors
	try:
		C = Tensor(xp.tensordot(tensor1.data, tensor2.data,
								(tensor1_indices, tensor2_indices)))
	except ValueError as e:
		# Print more useful info in case of ValueError.
		# Check if number of indices are equal
		if not len(tensor1_indices) == len(tensor2_indices):
			raise ValueError('Number of indices in contraction '
					'does not match.')
		# Check if indices have equal dimensions
		for i in range(len(tensor1_indices)):
			d1 = tensor1.data.shape[tensor1_indices[i]]
			d2 = tensor2.data.shape[tensor2_indices[i]]
			if d1 != d2:
				raise ValueError(labels1[i] + ' with dim=' + str(d1) +
									   ' does not match ' + labels2[i] +
									   ' with dim=' + str(d2))
		# Check if indices exist
		for i in range(len(labels1)):
			if not labels1[i] in tensor1.labels:
				raise ValueError(labels1[i] + 
						' not in list of labels for tensor1')
			if not labels2[i] in tensor2.labels:
				raise ValueError(labels2[i] + 
						' not in list of labels for tensor2')
		# Re-raise exception
		raise e

	# The following removes the contracted indices from the list of labels
	# and concatenates them
	new_tensor1_labels = [i for j, i in enumerate(tensor1.labels)
						  if j not in tensor1_indices]
	new_tensor2_labels = [i for j, i in enumerate(tensor2.labels)
						  if j not in tensor2_indices]
	C.labels = new_tensor1_labels + new_tensor2_labels

	return C


def tensor_product(tensor1, tensor2):
	"""Take tensor product of two tensors without contracting any indices"""
	return contract(tensor1, tensor2, [], [])



def directSumTensor(tensor1, tensor2, tensor1VirtLabels, tensor2VirtLabels, tensor1PhysLabels, tensor2PhysLabels):
	lenVirtLabel = len(tensor1VirtLabels)
	if lenVirtLabel != len(tensor2VirtLabels):
		raise ValueError("len(tensor1VirtLabels) != len(tensor2VirtLabels)")
	lenPhysLabel = len(tensor1PhysLabels)
	if lenPhysLabel != len(tensor2PhysLabels):
		raise ValueError("len(tensor1PhysLabels) != len(tensor2PhysLabels)")
	if lenVirtLabel+lenPhysLabel != tensor1.rank:
		raise ValueError("lenVirtLabel+lenPhysLabel != tensor1.rank")
	if lenVirtLabel+lenPhysLabel != tensor2.rank:
		raise ValueError("lenVirtLabel+lenPhysLabel != tensor2.rank")

	tensor1 = tensor1.copy()
	tensor2 = tensor2.copy()

	tensor1.move_indices_to_top(tensor1VirtLabels+tensor1PhysLabels)
	tensor2.move_indices_to_top(tensor2VirtLabels+tensor2PhysLabels)

	#tensor1VirtDims = [tensor1.dimension_of_label(label) for label in tensor1VirtLabels]
	#tensor2VirtDims = [tensor2.dimension_of_label(label) for label in tensor2VirtLabels]

	tensor1VirtPads = [(0,tensor2.dimension_of_label(label)) for label in tensor2VirtLabels]
	tensor2VirtPads = [(tensor1.dimension_of_label(label),0) for label in tensor1VirtLabels]

	tensor1.pad_indices(tensor1VirtLabels, tensor1VirtPads)
	tensor2.pad_indices(tensor2VirtLabels, tensor2VirtPads)

	tensor1 = tensor1.__iadd__(tensor2, True)
	return tensor1



def tensor_to_matrix(tensor, row_labels, column_labels=None):
	"""
	Convert a tensor to a matrix regarding row_labels as row index (output)
	and the remaining indices as column index (input).
	"""
	t = tensor

	t.move_indices_to_top(row_labels)
	if not column_labels is None:
		t.move_indices_to_bottom(column_labels)

	total_row_dimension = numpy.prod(t.data.shape[:len(row_labels)], dtype=int)
	total_column_dimension = numpy.prod(t.data.shape[len(row_labels):], dtype=int)


	return xp.reshape(t.data, (total_row_dimension, total_column_dimension))


def matrix_to_tensor(matrix, shape, labels=None):
	"""
	Convert a matrix to a tensor by reshaping to `shape` and giving labels
	specifid by `labels`
	"""
	temp = Tensor(matrix, labels=["temp_label_in_matrix_to_tensor"])
	temp.split_index("temp_label_in_matrix_to_tensor", shape, labels)
	return temp


def tensor_to_vector(tensor, row_labels):
	"""
	Convert a tensor to a matrix regarding row_labels as row index (output)
	and the remaining indices as column index (input).
	"""
	t = tensor

	t.move_indices_to_top(row_labels)
	total_row_dimension = numpy.prod(t.data.shape[:len(row_labels)], dtype=int)
	#print("total_row_dimension", total_row_dimension)

	return xp.reshape(t.data, (total_row_dimension, ))


def vector_to_tensor(vector, shape, labels=None):
	"""
	Convert a matrix to a tensor by reshaping to `shape` and giving labels
	specifid by `labels`
	"""
	if len(vector.shape)==2: vector = vector[:,0]
	temp = Tensor(vector, labels=["temp_label_in_vector_to_tensor"])
	temp.split_index("temp_label_in_vector_to_tensor", shape, labels)
	return temp



def tensor_svd(tensor, row_labels, svd_label="svd_",
			   absorb_singular_values=None):
	"""
	Compute the singular value decomposition of `tensor` after reshaping it 
	into a matrix.

	Indices with labels in `row_labels` are fused to form a single index 
	corresponding to the rows of the matrix (typically the left index of a
	matrix). The remaining indices are fused to form the column indices. An SVD
	is performed on this matrix, yielding three matrices u, s, v, where u and
	v are unitary and s is diagonal with positive entries. These three
	matrices are then reshaped into tensors U, S, and V. Contracting U, S and V
	together along the indices labelled by `svd_label` will yeild the original
	input `tensor`.

	Parameters
	----------
	tensor : Tensor
		The tensor on which the SVD will be performed.
	row_labels : list
		List of labels specifying the indices of `tensor` which will form the
		rows of the matrix on which the SVD will be performed.
	svd_label : str
		Base label for the indices that are contracted with `S`, the tensor of
		singular values. 
	absorb_singular_values : str, optional
		If "left", "right" or "both", singular values will be absorbed into
		U, V, or the square root into both, respectively, and only U and V
		are returned.

	Returns
	-------
	U : Tensor
		Tensor obtained by reshaping the matrix u obtained by SVD as described 
		above. Has indices labelled by `row_labels` corresponding to the
		indices labelled `row_labels` of `tensor` and has one index labelled 
		`svd_label`+"in" which connects to S.
	V : Tensor
		Tensor obtained by reshaping the matrix v obtained by SVD as described 
		above. Indices correspond to the indices of `tensor` that aren't in 
		`row_labels`. Has one index labelled  `svd_label`+"out" which connects
		to S.
	S : Tensor
		Tensor with data consisting of a diagonal matrix of singular values.
		Has two indices labelled `svd_label`+"out" and `svd_label`+"in" which
		are contracted with with the `svd_label`+"in" label of U and the
		`svd_label`+"out" of V respectively.

	Examples
	--------
	>>> a=random_tensor(2,3,4, labels = ["i0", "i1", "i2"])
	>>> U,S,V = tensor_svd(a, ["i0", "i2"])
	>>> print(U)
	Tensor object: shape = (2, 4, 3), labels = ['i0', 'i2', 'svd_in']
	>>> print(V)
	Tensor object: shape = (3, 3), labels = ['svd_out', 'i1']
	>>> print(S)
	Tensor object: shape = (3, 3), labels = ['svd_out', 'svd_in']
	
	Recombining the three tensors obtained from SVD, yeilds a tensor very close
	to the original.

	>>> temp=tn.contract(S, V, "svd_in", "svd_out")
	>>> b=tn.contract(U, temp, "svd_in", "svd_out")
	>>> tn.distance(a,b)
	1.922161284937472e-15
	"""

	t = tensor.copy()

	# Move labels in row_labels to the beginning of list, and reshape data
	# accordingly
	t.move_indices_to_top(row_labels)
	total_input_dimension = numpy.prod(t.data.shape[:len(row_labels)], dtype=int)

	column_labels = t.labels[len(row_labels):]

	old_shape = t.data.shape
	total_output_dimension = numpy.prod(t.data.shape[len(row_labels):], dtype=int)
	data_matrix = xp.reshape(t.data, (total_input_dimension,
									  total_output_dimension))

	try:
		u, s, v = xp.linalg.svd(data_matrix, full_matrices=False)
	except (xp.linalg.LinAlgError, ValueError):
		# Try with different lapack driver
		warnings.warn(('numpy.linalg.svd failed, trying scipy.linalg.svd with' +
					   ' lapack_driver="gesvd"'))
		try:
			u, s, v = xp.linalg.svd(data_matrix, full_matrices=False,
									lapack_driver='gesvd')
		except ValueError:
			# Check for inf's and nan's:
			print("tensor_svd failed. Matrix contains inf's: "
				  + str(xp.isinf(data_matrix).any())
				  + ". Matrix contains nan's: "
				  + str(xp.isnan(data_matrix).any()))
			raise  # re-raise the exception

	# New shape original index labels as well as svd index
	U_shape = list(old_shape[0:len(row_labels)])
	U_shape.append(u.shape[1])
	U = Tensor(data=xp.reshape(u, U_shape), labels=row_labels + [svd_label + "in"])
	V_shape = list(old_shape)[len(row_labels):]
	V_shape.insert(0, v.shape[0])
	V = Tensor(data=xp.reshape(v, V_shape),
			   labels=[svd_label + "out"] + column_labels)

	S = Tensor(data=xp.diag(s), labels=[svd_label + "out", svd_label + "in"])

	# Absorb singular values S into either V or U
	# or take the square root of S and absorb into both
	if absorb_singular_values == "left":
		U_new = contract(U, S, ["svd_in"], ["svd_out"])
		V_new = V
		return U_new, V_new
	elif absorb_singular_values == "right":
		V_new = contract(S, V, ["svd_in"], ["svd_out"])
		U_new = U
		return U_new, V_new
	elif absorb_singular_values == "both":
		sqrtS = S.copy()
		sqrtS.data = xp.sqrt(sqrtS.data)
		U_new = contract(U, sqrtS, ["svd_in"], ["svd_out"])
		V_new = contract(sqrtS, V, ["svd_in"], ["svd_out"])
		return U_new, V_new
	else:
		return U, S, V


def tensor_qr(tensor, row_labels, qr_label="qr_"):
	"""
	Compute the QR decomposition of `tensor` after reshaping it into a matrix.
	Indices with labels in `row_labels` are fused to form a single index
	corresponding to the rows of the matrix (typically the left index of a
	matrix). The remaining indices are fused to form the column index. A QR
	decomposition is performed on this matrix, yielding two matrices q,r, where
	q and is a rectangular matrix with orthonormal columns and r is upper
	triangular. These two matrices are then reshaped into tensors Q and R.
	Contracting Q and R along the indices labelled `qr_label` will yeild the
	original input tensor `tensor`.

	Parameters
	----------
	tensor : Tensor
		The tensor on which the QR decomposition will be performed.
	row_labels : list
		List of labels specifying the indices of `tensor` which will form the
		rows of the matrix on which the QR will be performed.
	qr_label : str
		Base label for the indices that are contracted between `Q` and `R`.

	Returns
	-------
	Q : Tensor
		Tensor obtained by reshaping the matrix q obtained from QR
		decomposition.  Has indices labelled by `row_labels` corresponding to
		the indices labelled `row_labels` of `tensor` and has one index
		labelled `qr_label`+"in" which connects to `R`.
	R : Tensor
		Tensor obtained by reshaping the matrix r obtained by QR decomposition.
		Indices correspond to the indices of `tensor` that aren't in
		`row_labels`. Has one index labelled `qr_label`+"out" which connects
		to `Q`.

	Examples
	--------

	>>> from tncontract.tensor import *
	>>> t=random_tensor(2,3,4)
	>>> print(t)
	Tensor object: shape = (2, 3, 4), labels = ['i0', 'i1', 'i2']
	>>> Q,R = tensor_qr(t, ["i0", "i2"])
	>>> print(Q)
	Tensor object: shape = (2, 4, 3), labels = ['i0', 'i2', 'qr_in']
	>>> print(R)
	Tensor object: shape = (3, 3), labels = ['qr_out', 'i1']

	Recombining the two tensors obtained from `tensor_qr`, yeilds a tensor very
	close to the original

	>>> x = contract(Q, R, "qr_in", "qr_out")
	>>> print(x)
	Tensor object: shape = (2, 4, 3), labels = ['i0', 'i2', 'i1']
	>>> distance(x,t)
	9.7619164946377426e-16
	"""
	t = tensor.copy()

	if not isinstance(row_labels, list):
		# If row_labels is not a list, convert to list with a single entry
		# "row_labels"
		row_labels = [row_labels]

	# Move labels in row_labels to the beginning of list, and reshape data
	# accordingly
	t.move_indices_to_top(row_labels)

	# Compute the combined dimension of the row indices
	row_dimension = 1
	for i, label in enumerate(t.labels):
		if label not in row_labels:
			break
		row_dimension *= t.data.shape[i]

	column_labels = [x for x in t.labels if x not in row_labels]

	old_shape = t.data.shape
	total_output_dimension = int(numpy.product(t.data.shape, dtype=int) / row_dimension)
	data_matrix = xp.reshape(t.data, (row_dimension,
									  total_output_dimension))

	q, r = xp.linalg.qr(data_matrix, mode="reduced")

	# New shape original index labels as well as svd index
	Q_shape = list(old_shape[0:len(row_labels)])
	Q_shape.append(q.shape[1])
	Q = Tensor(data=xp.reshape(q, Q_shape), labels=row_labels + [qr_label + "in"])
	R_shape = list(old_shape)[len(row_labels):]
	R_shape.insert(0, r.shape[0])
	R = Tensor(data=xp.reshape(r, R_shape), labels=[qr_label + "out"] +
												   column_labels)

	return Q, R


def tensor_lq(tensor, row_labels, lq_label="lq_"):
	"""
	Compute the LQ decomposition of `tensor` after reshaping it into a matrix.
	Indices with labels in `row_labels` are fused to form a single index
	corresponding to the rows of the matrix (typically the left index of a
	matrix). The remaining indices are fused to form the column index. An LR
	decomposition is performed on this matrix, yielding two matrices l,q, where
	q and is a rectangular matrix with orthonormal rows and l is upper
	triangular. These two matrices are then reshaped into tensors L and Q.
	Contracting L and Q along the indices labelled `lq_label` will yeild the
	original input `tensor`. Note that the LQ decomposition is actually
	identical to the QR decomposition after a relabelling of indices. 

	Parameters
	----------
	tensor : Tensor
		The tensor on which the LQ decomposition will be performed.
	row_labels : list
		List of labels specifying the indices of `tensor` which will form the
		rows of the matrix on which the LQ decomposition will be performed.
	lq_label : str
		Base label for the indices that are contracted between `L` and `Q`.

	Returns
	-------
	Q : Tensor
		Tensor obtained by reshaping the matrix q obtained by LQ decomposition.
		Indices correspond to the indices of `tensor` that aren't in
		`row_labels`. Has one index labelled `lq_label`+"out" which connects
		to `L`.
	L : Tensor
		Tensor obtained by reshaping the matrix l obtained from LQ
		decomposition.  Has indices labelled by `row_labels` corresponding to
		the indices labelled `row_labels` of `tensor` and has one index
		labelled `lq_label`+"in" which connects to `Q`.

	See Also
	--------
	tensor_qr
	
	"""

	col_labels = [x for x in tensor.labels if x not in row_labels]

	temp_label = lbl.unique_label()
	# Note the LQ is essentially equivalent to a QR decomposition, only labels
	# are renamed
	Q, L = tensor_qr(tensor, col_labels, qr_label=temp_label)
	Q.replace_label(temp_label + "in", lq_label + "out")
	L.replace_label(temp_label + "out", lq_label + "in")

	return L, Q


def truncated_svd(tensor, row_labels, chi=None, threshold=1e-15,
				  absorb_singular_values="right", absolute = True, svd_label="svd_"):
	"""
	Will perform svd of a tensor, as in tensor_svd, and provide approximate
	decomposition by truncating all but the largest k singular values then
	absorbing S into U, V or both. Truncation is performedby specifying the
	parameter chi (number of singular values to keep).

	Parameters
	----------
	chi : int, optional
		Maximum number of singular values of each tensor to keep after
		performing singular-value decomposition.
	threshold : float
		Threshold for the magnitude of singular values to keep.
		If absolute then singular values which are less than threshold will be truncated.
		If relative then singular values which are less than max(singular_values)*threshold will be truncated
	"""

	U, S, V = tensor_svd(tensor, row_labels, svd_label=svd_label)

	singular_values = xp.diag(S.data)

	# Truncate to maximum number of singular values

	if chi:
		singular_values_to_keep = singular_values[:chi]
		truncated_evals_1 = singular_values[chi:]
	else:
		singular_values_to_keep = singular_values
		truncated_evals_1 = xp.array([])

	# Thresholding

	if absolute:
		truncated_evals_2 = singular_values_to_keep[singular_values_to_keep <= threshold]
		singular_values_to_keep = singular_values_to_keep[singular_values_to_keep > threshold]
	else:
		rel_thresh = singular_values[0]*threshold
		truncated_evals_2 = singular_values_to_keep[singular_values_to_keep <= rel_thresh]
		singular_values_to_keep = singular_values_to_keep[singular_values_to_keep > rel_thresh]

	truncated_evals = xp.concatenate((truncated_evals_2, truncated_evals_1), axis=0)

	# Reconstitute and truncate corresponding singular index of U and V

	S.data = xp.diag(singular_values_to_keep)

	U.move_index_to_top(svd_label+"in")
	U.data = U.data[0:len(singular_values_to_keep)]
	U.move_index_to_bottom(svd_label+"in")
	V.data = V.data[0:len(singular_values_to_keep)]


	if absorb_singular_values is None:
		return U, S, V
	# Absorb singular values S into either V or U
	# or take the square root of S and absorb into both (default)
	if absorb_singular_values == "left":
		U_new = contract(U, S, [svd_label+"in"], [svd_label+"out"])
		V_new = V
	elif absorb_singular_values == "right":
		V_new = contract(S, V, [svd_label+"in"], [svd_label+"out"])
		U_new = U
	else:
		sqrtS = S.copy()
		sqrtS.data = xp.sqrt(sqrtS.data)
		U_new = contract(U, sqrtS, [svd_label+"in"], [svd_label+"out"])
		V_new = contract(sqrtS, V, [svd_label+"in"], [svd_label+"out"])

	return U_new, V_new, truncated_evals

