from __future__ import (absolute_import, division,
						print_function, unicode_literals)

"""
onedim_core
==========

Core module for onedimensional tensor networks
"""

#__all__ = ['OneDimensionalTensorNetwork', 'MatrixProductState', 'MatrixProductOperator', 
#		   'contract_mps_mpo', "contract_mpo_mpo", 
#		   'inner_product_mps', 'ladder_contract',
#		   ]

import math

from tncontract import tensor_core as tnc
from tncontract import tensor_instant as tni
from tncontract import label as lbl
from tncontract.onedim import onedim_instant as odi

class OneDimensionalTensorNetwork:
	"""
	A one-dimensional tensor network. MatrixProductState and
	MatrixProductOperator are subclasses of this class. 

	An instance of `OneDimensionalTensorNetwork` contains a one-dimensional
	array of tensors in its `data` attribute. This one dimensional array is
	specified in the `tensors` argument when initialising the array. Each
	tensor in `data` requires a left index and a right index. The right index
	is taken to be contracted with the left index of the next tensor in the
	array, while the left index is taken to be contracted with the right index
	of the previous tensor in the array. All left indices are assumed to have
	the same label, and likewise for the right indices. They are specified in
	the initialisation of array (by default they are assumed to be "left" and
	"right" respectively) and will be stored in the attributes `left_label` and
	`right_label` of the OneDimensionalTensorNetwork instance. 
	
	"""

	def __init__(self, tensors, left_label="left", right_label="right", boundaryType="O"):
		self.left_label = left_label
		self.right_label = right_label

		self.data = tensors

		# Every tensor will have three indices corresponding to "left", "right"
		# and "phys" labels. If only two are specified for left and right
		# boundary tensors (for open boundary conditions) an extra dummy index
		# of dimension 1 will be added.
		for x in self.data:
			if left_label not in x.labels: x.add_dummy_index(left_label)
			if right_label not in x.labels: x.add_dummy_index(right_label)

		#bondaryType == "O" means open boundary, "P" means periodic boundary, "S" means slice of another OTN
		self.boundaryType = boundaryType



	def __iter__(self):
		return self.data.__iter__()

	def __len__(self):
		return self.data.__len__()

	def __getitem__(self, key):
		return self.data.__getitem__(key)

	def __setitem__(self, key, value):
		self.data.__setitem__(key, value)



	def leftDim(self, site):
		"""Return left index dimesion for site"""
		return self.data[site].index_dimension(self.left_label)

	def rightDim(self, site):
		"""Return right index dimesion for site"""
		return self.data[site].index_dimension(self.right_label)

	def bondDims(self):
		"""Return list of all bond dimensions"""
		if self.nsites == 0:
			return []
		bonds = [self.leftDim(0)]
		for i in range(self.nsites):
			bonds.append(self.rightDim(i))
		return bonds

	@property
	def nsites(self):
		return len(self.data)

	@property
	def nsites_physical(self):
		return self.nsites



	def copy(self):
		"""Alternative the standard copy method, returning a
		OneDimensionalTensorNetwork that is not
		linked in memory to the previous ones."""
		return OneDimensionalTensorNetwork([x.copy() for x in self],
										   self.left_label, self.right_label, self.boundaryType)

	def conjugate(self):
		for x in self.data:
			x.conjugate()

	def conjugated(self):
		temp = self.copy()
		temp.conjugate()
		return temp



	def reverse(self):
		self.data = self.data[::-1]
		temp = self.left_label
		self.left_label = self.right_label
		self.right_label = temp

	def reversed(self):
		temp = self.copy()
		temp.reverse()
		return temp

	def leftshift(self, x):
		self.data = self.data[x:]+self.data[:x]

	def leftshifted(self,x):
		temp = self.copy()
		temp.leftshift(x)
		return temp

	def rightshift(self, x):
		self.data = self.data[-x:]+self.data[:-x]

	def rightshifted(self,x):
		temp = self.copy()
		temp.rightshift(x)
		return temp

	def deepsliced(self,start,end):
		return OneDimensionalTensorNetwork([self[i].copy() for i in range(start,end)],
										   self.left_label, self.right_label, "S")

	def sliced(self,start,end):
		return OneDimensionalTensorNetwork(self[start:end], self.left_label, self.right_label, "S")

	def replace_left_label(self, new_label):
		old_label = self.left_label
		if old_label!=new_label:
			self.left_label = new_label
			for x in self.data:
				x.replace_label(old_label, new_label)

	def replace_right_label(self, new_label):
		old_label = self.right_label
		if old_label!=new_label:
			self.right_label = new_label
			for x in self.data:
				x.replace_label(old_label, new_label)

	def replace_labels(self, old_labels, new_labels):
		"""Run `Tensor.replace_label` method on every tensor in `self` then
		replace `self.left_label` and `self.right_label` appropriately."""

		if not isinstance(old_labels, list):
			old_labels = [old_labels]
		if not isinstance(new_labels, list):
			new_labels = [new_labels]

		for x in self.data:
			x.replace_label(old_labels, new_labels)

		if self.left_label in old_labels:
			self.left_label = new_labels[old_labels.index(self.left_label)]
		if self.right_label in old_labels:
			self.right_label = new_labels[old_labels.index(self.right_label)]

	def standard_virtual_labels(self, suffix=""):
		"""Replace `self.left_label` with "left"+`suffix` and 
		`self.right_label` with "right"+`suffix`."""

		self.replace_labels([self.left_label, self.right_label],
							["left" + suffix, "right" + suffix])

	def unique_virtual_labels(self):
		"""Replace `self.left_label` and `self.right_label` with unique labels
		generated by tensor.lbl.unique_label()."""

		self.replace_labels([self.left_label, self.right_label],
							[lbl.unique_label(), lbl.unique_label()])



	def __imul__(self, scalar):
		if isinstance(scalar, float) or isinstance(scalar, int) or isinstance(scalar, complex):
			self.data[0] = self.data[0]*scalar
			return self
		else:
			return NotImplemented

	def __mul__(self, scalar):
		if isinstance(scalar, float) or isinstance(scalar, int) or isinstance(scalar, complex):
			temp = self.copy()
			temp *= scalar
			return temp
		else:
			return NotImplemented

	def __rmul__(self, scalar):
		if isinstance(scalar, float) or isinstance(scalar, int) or isinstance(scalar, complex):
			temp = self.copy()
			temp *= scalar
			return temp
		else:
			return NotImplemented


	def normaliseAllData(self): #!!!UNYEDITTED!!!
		for tensor in self.data:
			tensor.normalise()



class MatrixProductState(OneDimensionalTensorNetwork):
	"""Matrix product state"is a list of tensors, each having and index 
	labelled "phys" and at least one of the indices "left", "right"
	Input is a list of tensors, with three up to three index labels, If the 
	labels aren't already specified as "left", "right", "phys" need to specify
	which labels correspond to these using arguments left_label, right_label, 
	phys_label. The tensors input will be copied, and will not point in memory
	to the original ones."""

	def __init__(self, tensors, left_label="left", right_label="right",
				 phys_label="phys", isKet=True, left_canonised_up_to=None, right_canonised_up_to=None, boundaryType="O"):
		OneDimensionalTensorNetwork.__init__(self, tensors,
											 left_label=left_label, right_label=right_label, boundaryType=boundaryType)
		self.phys_label = phys_label
		self.isKet = isKet
		if left_canonised_up_to is None:
			self.left_canonised_up_to = 0
		else:
			self.left_canonised_up_to = left_canonised_up_to
		if right_canonised_up_to is None:
			self.right_canonised_up_to = len(self)
		else:
			self.right_canonised_up_to = right_canonised_up_to

	def __repr__(self):
		return ("MatrixProductState(tensors=%r, left_label=%r, right_label=%r,"
				"phys_label=%r, isKet=%r, left_canonised_up_to=%r, right_canonised_up_to=%r)"
				% (self.data, self.left_label, self.right_label, self.phys_label
					, self.isKet, self.left_canonised_up_to, self.right_canonised_up_to))

	def __str__(self):
		return ("MatrixProductState object: " +
				"sites = " + str(len(self)) +
				", left_label = " + self.left_label +
				", right_label = " + self.right_label +
				", phys_label = " + self.phys_label +
				", isKet = " + str(self.isKet) +
				", left_canonised_up_to = " + str(self.left_canonised_up_to) +
				", right_canonised_up_to = " + str(self.right_canonised_up_to)
				)

	def __setitem__(self, key, value):
		self.data.__setitem__(key, value)
		self.left_canonised_up_to = min(self.left_canonised_up_to, key)
		self.right_canonised_up_to = max(self.right_canonised_up_to, key+1)


	def physical_site(self, n):
		""" Return position of n'th physical (pos=n). Implemented for 
		comaptibility with MatrixProductStateCanonical."""
		return n

	def physDim(self, site):
		"""Return physical index dimesion for site"""
		return self.data[site].index_dimension(self.phys_label)

	physoutDim = physDim
	physinDim = physDim

	def physDims(self):
		res = []
		for i in range(self.nsites):
			res.append(self.physDim(i))
		return res



	def copy(self):
		"""Return an MPS that is not linked in memory to the original."""
		return MatrixProductState([x.copy() for x in self], self.left_label, self.right_label, self.phys_label, self.isKet, self.left_canonised_up_to, self.right_canonised_up_to)

	def transpose(self):
		self.isKet=not(self.isKet)

	def transposed(self):
		temp = self.copy()
		temp.transpose()
		return temp

	def adjoint(self):
		self.transpose()
		self.conjugate()

	def adjointed(self):
		temp = self.copy()
		temp.adjoint()
		return temp


	def reverse(self):
		self.data = self.data[::-1]
		temp = self.left_label
		self.left_label = self.right_label
		self.right_label = temp
		temp = len(self) - self.left_canonised_up_to
		self.left_canonised_up_to = len(self) - self.right_canonised_up_to
		self.right_canonised_up_to = temp

	def reversed(self):
		temp = self.copy()
		temp.reverse()
		return temp


	def deepsliced(self,start,end):
		return MatrixProductState([self[i].copy() for i in range(start,end)],
				left_label=self.left_label, right_label=self.right_label,
				phys_label=self.phys_label, isKet=self.isKet, 
				left_canonised_up_to=max(0, self.left_canonised_up_to-start),
				right_canonised_up_to=min(end, self.right_canonised_up_to)-start,
				boundaryType="S")

	def sliced(self,start,end):
		return MatrixProductState(self[start:end],
				left_label=self.left_label, right_label=self.right_label,
				phys_label=self.phys_label, isKet=self.isKet, 
				left_canonised_up_to=max(0, self.left_canonised_up_to-start),
				right_canonised_up_to=min(end, self.right_canonised_up_to)-start,
				boundaryType="S")


	def replace_phys_label(self, new_label):
		old_label = self.phys_label
		if old_label!=new_label:
			self.phys_label = new_label
			for x in self.data:
				x.replace_label(old_label, new_label)

	def replace_labels(self, old_labels, new_labels):
		"""run `tensor.replace_label` method on every tensor in `self` then
		replace `self.left_label`, `self.right_label` and `self.phys_label` 
		appropriately."""

		if not isinstance(old_labels, list):
			old_labels = [old_labels]
		if not isinstance(new_labels, list):
			new_labels = [new_labels]

		for x in self.data:
			x.replace_label(old_labels, new_labels)

		if self.left_label in old_labels:
			self.left_label = new_labels[old_labels.index(self.left_label)]
		if self.right_label in old_labels:
			self.right_label = new_labels[old_labels.index(self.right_label)]
		if self.phys_label in old_labels:
			self.phys_label = new_labels[old_labels.index(self.phys_label)]

	def standard_labels(self, suffix=""):
		"""
		overwrite self.labels, self.left_label, self.right_label, 
		self.phys_label with standard labels "left", "right", "phys"
		"""
		self.replace_labels([self.left_label, self.right_label, self.phys_label],
							["left" + suffix, "right" + suffix, "phys" + suffix])

	def unique_labels(self): #!!!UNYEDITTED!!!
		self.replace_labels([self.left_label, self.right_label, self.phys_label],
							[lbl.unique_label(), lbl.unique_label(), lbl.unique_label()])



	def __mul__(self,other):
		if isinstance(other, MatrixProductOperator):
			return contract_mps_mpo_to_mps(self, other)
		elif isinstance(other, MatrixProductState):
			return contract_mps_mps_to_scalar(self, other)
		else:
			return OneDimensionalTensorNetwork.__mul__(self, other)

	def __rmul__(self,other):
		if isinstance(other, MatrixProductOperator):
			return contract_mpo_mps_to_mps(other, self)
		else:
			return OneDimensionalTensorNetwork.__rmul__(self, other)

	def __add__(self, other):
		if isinstance(other, MatrixProductState):
			return directSum_mps_mps_to_mps(self, other)
		else:
			return NotImplemented


	def norm(self, canonical_form=None):
		#return sqrt(inner_product_mps(self[self.left_canonised_up_to, self.right_canonised_up_to], self[self.left_canonised_up_to, self.right_canonised_up_to]))
		"""Return norm of mps.

		Parameters
		----------

		canonical_form : str
			If `canonical_form` is "left", the state will be assumed to be in
			left canonical form, if "right" the state will be assumed to be in
			right canonical form. In these cases the norm can be read off the 
			last tensor (much more efficient). 
		"""

		if canonical_form == "left":
			return self[-1].norm()
		elif canonical_form == "right":
			return self[0].data.norm()
		else:
			return math.sqrt( complex(inner_product_mps(self, self)).real )

	def kitaichi(self, mpo):
		if self.isKet:
			ket = self
			bra = self.adjointed()
		else:
			bra = self
			ket = self.adjointed()
		#print("bra",bra)
		#print("ket",ket)
		psiPsi = bra * ket
		#print("psiPsi",psiPsi)
		#print("mpo*ket", mpo*ket)
		psiOPsi = bra * (mpo * ket)
		#print("psiOPsi",psiOPsi)
		return psiOPsi/psiPsi

	def normalise(self):
		norm = self.norm()
		if norm <= 1e-10: return 0
		kakeru = norm**(-1/len(self))
		for site in range(len(self)):
			self[site] = self[site]*kakeru


	def left_canonise_site(self, focusingSite, chi=None, threshold=1e-14,
					  normalise=False, qr_decomposition=False):
		#print("called left_canonised_site("+str(focusingSite)+")")
		i=focusingSite

		if i == len(self)-1:
			norm = self[i].norm()
			if norm==0.0:
				#reisei ni kangaete i, i+1 igai ga kawaruno kimoi node kou kaemasu
				self[i] = tni.zeros_tensor_like(self[i])
				return
			if normalise == True:
				self[i].data = self[i].data / norm
			return


		if qr_decomposition:
			qr_label = lbl.unique_label()
			Q, R = tnc.tensor_qr(self[i], [self.phys_label,
											   self.left_label], qr_label=qr_label)

			# Replace tensor at site i with Q
			Q.replace_label(qr_label + "in", self.right_label)
			self[i] = Q

			# Absorb R into next tensor
			self[i + 1] = tnc.contract(R, self[i + 1], self.right_label,
									   self.left_label)
			self[i + 1].replace_label(qr_label + "out", self.left_label)

		else:
			svd_label = lbl.unique_label()
			U, V, _ = tnc.truncated_svd(self[i], [self.phys_label, self.left_label], chi=chi, threshold=threshold, absorb_singular_values="right", absolute=False, svd_label=svd_label)

			U.replace_label(svd_label + "in", self.right_label)
			self[i] = U
			self[i + 1] = tnc.contract(V, self[i + 1], self.right_label,
									   self.left_label)
			self[i + 1].replace_label(svd_label + "out", self.left_label)

		if self.left_canonised_up_to >= focusingSite:
			self.left_canonised_up_to = focusingSite+1

	def left_canonise_up_to(self, to_pun, chi=None, threshold=1e-14,
					  normalise=False, qr_decomposition=False):
		from_pun = self.left_canonised_up_to
		for site in range(from_pun, to_pun):
			#print("canonising", site)
			self.left_canonise_site(site, chi=chi, threshold=threshold,
					  normalise=normalise, qr_decomposition=qr_decomposition)
		self.left_canonised_up_to = max(from_pun, to_pun)
		return from_pun

	def right_canonise_up_to(self, to_pun, chi=None, threshold=1e-14,
					  normalise=False, qr_decomposition=False):
		from_pun = self.right_canonised_up_to
		self.reverse()
		self.left_canonise_up_to(len(self)-to_pun, chi=chi, threshold=threshold, normalise=normalise, qr_decomposition=qr_decomposition)
		self.reverse()
		return from_pun





class MatrixProductStateCanonical:
	pass



class MatrixProductOperator(OneDimensionalTensorNetwork):
	# TODO currently assumes open boundaries
	def __init__(self, tensors, left_label="left", right_label="right",
				 physout_label="physout", physin_label="physin", offset=0, boundaryType="O"):
		OneDimensionalTensorNetwork.__init__(self, tensors, left_label, right_label, boundaryType=boundaryType)
		self.physout_label = physout_label
		self.physin_label = physin_label
		self.offset=offset

	def __repr__(self):
		return ("MatrixProductOperator(tensors=%r, left_label=%r,"
				" right_label=%r, physout_label=%r, phsin_labe=%r)"
				% (self.data, self.left_label, self.right_label,
				   self.physout_label, self.physin_label))

	def __str__(self):
		return ("MatrixProductOperator object: " +
				"sites = " + str(len(self)) +
				", left_label = " + self.left_label +
				", right_label = " + self.right_label +
				", physout_label = " + self.physout_label +
				", physin_label = " + self.physin_label)



	def physoutDim(self, site):
		return self.data[site].index_dimension(self.physout_label)

	def physinDim(self, site):
		return self.data[site].index_dimension(self.physin_label)

	#def physDim(self, site):
	#	temp = self.physoutDim(site)
	#	if temp != self.physinDim(site):
	#		raise Exception("physinDim != physoutDim")
	#	return temp
	physDim = physoutDim

	def physDims(self):
		res = []
		for i in range(self.nsites):
			res.append(self.physDim(i))
		return res



	def copy(self):
		return MatrixProductOperator([x.copy() for x in self], self.left_label,
								  self.right_label, self.physout_label, self.physin_label, self.offset)
	def transpose(self):
		temp = self.physout_label
		self.physout_label = self.physin_label
		self.physin_label = temp

	def transposed(self):
		temp = self.copy()
		temp.transpose()
		return temp

	def adjoint(self):
		self.conjugate()
		self.transpose()

	def adjointed(self):
		temp = self.copy()
		temp.adjoint()
		return temp



	def deepsliced(self,start,end):
		return MatrixProductOperator([self[i].copy() for i in range(start,end)],
				left_label=self.left_label, right_label=self.right_label,
				physout_label=self.physout_label, physin_label=self.physin_label, 
				offset = self.offset - start,
				boundaryType="S")

	def sliced(self,start,end):
		return MatrixProductOperator(self[start:end],
				left_label=self.left_label, right_label=self.right_label,
				physout_label=self.physout_label, physin_label=self.physin_label, 
				offset = self.offset - start,
				boundaryType="S")


	def replace_physout_label(self, new_label):
		old_label = self.physout_label
		if old_label!=new_label:
			self.physout_label = new_label
			for x in self.data:
				x.replace_label(old_label, new_label)

	def replace_physin_label(self, new_label):
		old_label = self.physin_label
		if old_label!=new_label:
			self.physin_label = new_label
			for x in self.data:
				x.replace_label(old_label, new_label)

	def replace_labels(self, old_labels, new_labels):
		"""run `tensor.replace_label` method on every tensor in `self` then
		replace `self.left_label`, `self.right_label` and `self.phys_label` 
		appropriately."""

		if not isinstance(old_labels, list):
			old_labels = [old_labels]
		if not isinstance(new_labels, list):
			new_labels = [new_labels]

		for x in self.data:
			x.replace_label(old_labels, new_labels)

		if self.left_label in old_labels:
			self.left_label = new_labels[old_labels.index(self.left_label)]
		if self.right_label in old_labels:
			self.right_label = new_labels[old_labels.index(self.right_label)]
		if self.physout_label in old_labels:
			self.physout_label = new_labels[old_labels.index(self.physout_label)]
		if self.physin_label in old_labels:
			self.physin_label = new_labels[old_labels.index(self.physin_label)]


	def unique_labels(self):
		self.replace_labels([self.left_label, self.right_label, self.physout_label, self.physin_label],
							[lbl.unique_label(), lbl.unique_label(), lbl.unique_label(), lbl.unique_label()])




	def __mul__(self,other):
		if isinstance(other, MatrixProductOperator):
			return contract_mpo_mpo_to_mpo(self, other)
		else:
			return OneDimensionalTensorNetwork.__mul__(self, other)

	def __add__(self, other):
		if isinstance(other, MatrixProductOperator):
			return directSum_mpo_mpo_to_mpo(self, other)
		else:
			return NotImplemented

	def trace(self):
		tempList = self.copy().data
		for tensor in tempList:
			tensor.trace(self.physout_label, self.physin_label)
		ansTensor = tempList[0]
		for tensor in tempList[1:]:
			ansTensor.contract(tensor, [self.right_label], [self.left_label])
		ansTensor.trace(self.right_label, self.left_label)
		return ansTensor.data


	def kitaichi(self, mpo):
		TrRho = (self).trace()
		#print("TrRho",TrRho)
		TrORho = (mpo * self).trace()
		#print("TrORho",TrORho)
		return TrORho/TrRho

	def kitaichiForcingHermite(self, mpo):
		#print("\n")
		TrRho = (self).trace()
		#print("TrRho",TrRho)
		TrORho = (mpo * self).trace()
		#print("TrORho",TrORho)
		adjSelf = self.adjointed()
		TrAdjRho = (adjSelf).trace()
		#print("TrAdjRho",TrAdjRho)
		TrOAdjRho = (mpo * adjSelf).trace()
		#print("TrOAdjRho",TrOAdjRho)
		return (TrORho+TrOAdjRho)/(TrRho+TrAdjRho)

	def kitaichiForcingAntiHermite(self, mpo):
		#print("\n")
		TrRho = (self).trace()
		#print("TrRho",TrRho)
		TrORho = (mpo * self).trace()
		#print("TrORho",TrORho)
		adjSelf = self.adjointed()
		TrAdjRho = (adjSelf).trace()
		#print("TrAdjRho",TrAdjRho)
		TrOAdjRho = (mpo * adjSelf).trace()
		#print("TrOAdjRho",TrOAdjRho)
		return (TrORho-TrOAdjRho)/(TrRho-TrAdjRho)

	def forceHermite_Site(self,site):
		tensor = self[site]
		tensor.forceHermite(self.physout_label,self.physin_label)
		self[site]=tensor


	def forceHermite(self):
		for site in range(len(self)):
			self.forceHermite_Site(site)




def ladacon(bra, ket, adjoint_bra=False, boundaryContractStyle="P"):
	if adjoint_bra:
		bra = bra.adjointed()
	else:
		bra = bra.copy()
	ket = ket.copy()

	bra.unique_labels()
	ket.unique_labels()

	for i in range(0, len(bra)):
		if i == 0:
			C = tnc.contract(bra[0], ket[0], bra.phys_label, ket.phys_label)
		else:
			C.contract(bra[i], bra.right_label, bra.left_label)
			C.contract(ket[i], [ket.right_label, bra.phys_label],
					   [ket.left_label, ket.phys_label])

	if boundaryContractStyle=="S":
		C.trace(bra.left_label, ket.left_label)
		C.trace(bra.right_label, ket.right_label)
	elif boundaryContractStyle=="P" or boundaryContractStyle=="O":
		C.trace(ket.left_label, ket.right_label)
		C.trace(bra.left_label, bra.right_label)

	return C


"""
def ladacon2(bra, ops, ket, adjoint_bra=False, boundaryContractStyle="P"):
	if adjoint_bra:
		bra = bra.adjointed()
	else:
		bra = bra.copy()
	ket = ket.copy()
	ops = [op.copy() for op in ops]
	opl = len(ops)

	bra.unique_labels()
	ket.unique_labels()
	for op in ops:
		op.unique_labels()

	for i in range(0, len(self)):
		if i == 0:
			C = tnc.contract(bra[0], ket[0], bra.phys_label, ket.phys_label)
		else:
			C.contract(bra[i], bra.right_label, bra.left_label)
			C.contract(ket[i], [ket.right_label, bra.phys_label],
					   [ket.left_label, ket.phys_label])

	if boundaryContractStyle=="S":
		C.trace(bra.left_label, ket.left_label)
		C.trace(bra.right_label, ket.right_label)
	elif boundaryContractStyle=="P" or boundaryContractStyle=="O":
		C.trace(ket.left_label, ket.right_label)
		C.trace(bra.left_label, bra.right_label)

	return C
"""





def ladder_contract(array1, array2, label1, label2, start=0, end=None,
					adjoint_array1=False, left_output_label="left",
					right_output_label="right", return_intermediate_contractions=False):
	"""
	Contract two one-dimensional tensor networks. Indices labelled `label1` in
	`array1` and indices labelled `label2` in `array2` are contracted pairwise
	and all virtual indices are contracted.  The contraction pattern
	resembles a ladder when represented graphically. 

	Parameters
	----------

	array1 : OneDimensionalTensorNetwork
	array2 : OneDimensionalTensorNetwork
		The one-dimensional networks to be contracted.

	label1 : str
	label2 : str
		The index labelled `label1` is contracted with the index labelled
		`label2` for every site in array.

	start : int
	end : int
		The endpoints of the interval to be contracted. The leftmost tensors
		involved in the contraction are `array1[start]` and `array2[start]`,
		while the rightmost tensors are `array2[end]` and `array2[end]`. 

	adjoint_array1 : bool
		Whether the complex adjoint of `array1` will be used, rather than
		`array1` itself. This is useful if, for instance, the two arrays are
		matrix product states and the inner product is to be taken (Note that
		inner_product_mps could be used in this case). 

	right_output_label : str
		Base label assigned to right-going indices of output tensor.
		Right-going indices will be assigned labels `right_output_label`+"1"
		and `right_output_label`+"2" corresponding, respectively, to `array1`
		and `array2`.

	left_output_label : str
		Base label assigned to left-going indices of output tensor. Left-going
		indices will be assigned labels `left_output_label`+"1" and
		`left_output_label`+"2" corresponding, respectively, to `array1` and
		`array2`.

	return_intermediate_contractions : bool
		If true, a list of tensors is returned. If the contraction is performed
		from left to right (see Notes below), the i-th entry contains the
		contraction up to the i-th contracted pair. If contraction is performed
		from right to left, this order is reversed (so the last entry
		corresponds to the contraction of the right-most pair tensors, which
		are first to be contracted).

	Returns
	-------
	tensor : Tensor
	   Tensor obtained by contracting the two arrays. The tensor may have left
	   indices, right indices, both or neither depending on the interval
	   specified. 

	intermediate_contractions : list 
		If `return_intermediate_contractions` is true a list
		`intermediate_contractions` is returned containing a list of tensors
		corresponding to contraction up to a particular column.

	Notes
	-----
	If the interval specified contains the left open boundary, contraction is
	performed from left to right. If not and if interval contains right
	boundary, contraction is performed from right to left. If the interval
	does not contain either boundary, contraction is performed from left to
	right.
	"""

	# If no end specified, will contract to end
	if end == None:
		end = min(array1.nsites, array2.nsites) - 1  # index of the last site

	if end < start:
		raise ValueError("Badly defined interval (end before start).")

	a1 = array1.copy()
	a2 = array2.copy()

	if adjoint_array1:
		a1.adjoint()

	# Give all contracted indices unique labels so no conflicts with other
	# labels in array1, array2
	a1.unique_virtual_labels()
	a2.unique_virtual_labels()
	rung_label = lbl.unique_label()
	a1.replace_labels(label1, rung_label)
	a2.replace_labels(label2, rung_label)

	intermediate_contractions = []
	if start == 0:  # Start contraction from left
		for i in range(0, end + 1):
			if i == 0:
				C = tnc.contract(a1[0], a2[0], rung_label, rung_label)
			else:
				C.contract(a1[i], a1.right_label, a1.left_label)
				C.contract(a2[i], [a2.right_label, rung_label],
						   [a2.left_label, rung_label])

			if return_intermediate_contractions:
				t = C.copy()
				t.replace_label([a1.right_label, a2.right_label],
								[right_output_label + "1", right_output_label + "2"])
				# Remove dummy indices except the right indices
				t.remove_all_dummy_indices(labels=[x for x in t.labels if x
												   not in [right_output_label + "1", right_output_label + "2"]])
				intermediate_contractions.append(t)

		C.replace_label([a1.right_label, a2.right_label],
						[right_output_label + "1", right_output_label + "2"])
		C.remove_all_dummy_indices()

	elif end == a1.nsites - 1 and end == a2.nsites - 1:  # Contract from the right
		for i in range(end, start - 1, -1):
			if i == end:
				C = tnc.contract(a1[end], a2[end], rung_label, rung_label)
			else:
				C.contract(a1[i], a1.left_label, a1.right_label)
				C.contract(a2[i], [a2.left_label, rung_label],
						   [a2.right_label, rung_label])

			if return_intermediate_contractions:
				t = C.copy()
				t.replace_label([a1.left_label, a2.left_label],
								[left_output_label + "1", left_output_label + "2"])
				# Remove dummy indices except the left indices
				t.remove_all_dummy_indices(labels=[x for x in t.labels if x
												   not in [left_output_label + "1", left_output_label + "2"]])
				intermediate_contractions.insert(0, t)

		C.replace_label([a1.left_label, a2.left_label],
						[left_output_label + "1", left_output_label + "2"])
		C.remove_all_dummy_indices()

	else:
		# When an interval does not contain a boundary, contract in pairs first
		# then together
		for i in range(start, end + 1):
			t = tnc.contract(a1[i], a2[i], rung_label, rung_label)
			if i == start:
				C = t
			else:
				C.contract(t, [a1.right_label, a2.right_label],
						   [a1.left_label, a2.left_label])

			if return_intermediate_contractions:
				t = C.copy()
				t.replace_label([a1.right_label, a2.right_label, a1.left_label,
								 a2.left_label], [right_output_label + "1",
												  right_output_label + "2", left_output_label + "1",
												  left_output_label + "2"])
				# Remove dummy indices except the left and right indices
				t.remove_all_dummy_indices(labels=[x for x in t.labels if x
												   not in [right_output_label + "1", right_output_label + "2",
														   left_output_label + "1", left_output_label + "2"]])
				t.remove_all_dummy_indices()
				intermediate_contractions.append(t)

		C.replace_label([a1.right_label, a2.right_label, a1.left_label,
						 a2.left_label], [right_output_label + "1", right_output_label + "2",
										  left_output_label + "1", left_output_label + "2"])
		C.remove_all_dummy_indices()

	if return_intermediate_contractions:
		return intermediate_contractions
	else:
		return C


def inner_product_mps(mps_bra, mps_ket, adjoint_bra=True,
					  return_whole_tensor=False):
	if isinstance(mps_bra, MatrixProductStateCanonical):
		mps_bra_tmp = canonical_to_left_canonical(mps_bra)
	else:
		mps_bra_tmp = mps_bra
	if isinstance(mps_ket, MatrixProductStateCanonical):
		mps_ket_tmp = canonical_to_left_canonical(mps_ket)
	else:
		mps_ket_tmp = mps_ket
	t = ladacon(mps_bra_tmp, mps_ket_tmp, adjoint_bra=adjoint_bra)
	return t.data
	if return_whole_tensor:
		return t
	else:
		return t.data


def contract_mps_mps_to_scalar(mps_bra, mps_ket, adjoint_bra=False):
	assert not(adjoint_bra) and not(mps_bra.isKet) or adjoint_bra and mps_bra.isKet, "contractedScalar_Mpo_Mps argument bra is assumed to be a bra, but this is a ket. ok?"
	assert mps_ket.isKet, "contractedScalar_Mpo_Mps argument ket is assumed to be a ket, but this is a bra. ok?"

	if isinstance(mps_bra, MatrixProductStateCanonical):
		mps_bra_tmp = canonical_to_left_canonical(mps_bra)
	else:
		mps_bra_tmp = mps_bra
	if isinstance(mps_ket, MatrixProductStateCanonical):
		mps_ket_tmp = canonical_to_left_canonical(mps_ket)
	else:
		mps_ket_tmp = mps_ket
	t = ladacon(mps_bra_tmp, mps_ket_tmp, adjoint_bra=adjoint_bra)
	return t.data


def contract_mpo_mps_to_mps(mpo, mps):
	assert mps.isKet, "contract_mpo_mps_to_mps argument mps is assumed to be a ket, but this is a bra. ok?"

	mpo = mpo.copy()
	mps = mps.copy()
	mpo.unique_labels()

	for mpoi in range(len(mpo)):
		mpsi = (mpoi + mpo.offset)%len(mps)
		mps[mpsi].contract(mpo[mpoi], mps.phys_label, mpo.physin_label)
		mps[mpsi].fuse_indices([mps.left_label, mpo.left_label], mps.left_label)
		mps[mpsi].fuse_indices([mps.right_label, mpo.right_label], mps.right_label)
		mps[mpsi].replace_label(mpo.physout_label, mps.phys_label)

	return mps


def contract_mps_mpo_to_mps(mps, mpo):
	assert not(mps.isKet), "contract_mps_mpo_to_mps argument mps is assumed to be a bra, but this is a ket. ok?"

	mps = mps.copy()
	mpo = mpo.copy()
	mpo.unique_labels()

	for mpoi in range(len(mpo)):
		mpsi = (mpoi + mpo.offset)%len(mps)
		mps[mpsi].contract(mpo[mpoi], mps.phys_label, mpo.physout_label)
		mps[mpsi].fuse_indices([mps.left_label, mpo.left_label], mps.left_label)
		mps[mpsi].fuse_indices([mps.right_label, mpo.right_label], mps.right_label)
		mps[mpsi].replace_label(mpo.physin_label, mps.phys_label)

	return mps


def contract_mpo_mpo_to_mpo(mpo1, mpo2, new_left_label="left_byCMpoMpoMpo", new_right_label="right_byCMpoMpoMpo", new_physout_label="physout_byCMpoMpoMpo", new_physin_label="physin_byCMpoMpoMpo"):
	mpo1 = mpo1.copy()
	mpo2 = mpo2.copy()

	newMpoOffset = min(mpo1.offset, mpo2.offset)
	newMpoLen = max(len(mpo1)+mpo1.offset, len(mpo2)+mpo2.offset) - newMpoOffset
	newMpo = odi.identity_mpo(newMpoLen, mpo1.physDim(0), left_label=new_left_label, right_label=new_right_label, physout_label=new_physout_label, physin_label=new_physin_label, boundaryType=mpo1.boundaryType, offset=newMpoOffset, dtype=mpo1[0].dtype)

	for mpo1i in range(len(mpo1)):
		newMpoi = mpo1i + mpo1.offset - newMpo.offset
		newMpo[newMpoi].contract(mpo1[mpo1i], new_physin_label, mpo1.physout_label)
		newMpo[newMpoi].fuse_indices([newMpo.left_label, mpo1.left_label], new_left_label)
		newMpo[newMpoi].fuse_indices([newMpo.right_label, mpo1.right_label], new_right_label) 
		newMpo[newMpoi].replace_label(mpo1.physin_label, new_physin_label)

	for mpo2i in range(len(mpo2)):
		newMpoi = mpo2i + mpo2.offset - newMpo.offset
		newMpo[newMpoi].contract(mpo2[mpo2i], new_physin_label, mpo2.physout_label)
		newMpo[newMpoi].fuse_indices([newMpo.left_label, mpo2.left_label], new_left_label)
		newMpo[newMpoi].fuse_indices([newMpo.right_label, mpo2.right_label], new_right_label) 
		newMpo[newMpoi].replace_label(mpo2.physin_label, new_physin_label)

	return newMpo




def directSum_mpo_mpo_to_mpo(mpo1, mpo2):
	assert mpo1.boundaryType==mpo2.boundaryType, "directSum_mpo_mpo_to_mpo assumes mpo1.boundaryType==mpo2.boundaryType, but\nmpo1.boundaryType == {0}\nmpo2.boundaryType == {1}\nok?".format(mpo1.boundaryType, mpo2.boundaryType)

	newTensors=[]
	for i,(tensor1,tensor2) in enumerate(zip(mpo1.data,mpo2.data)):
		if i==0 and i==len(mpo1)-1 and mpo1.boundaryType=="O":
			tensor1VirtLabels=[]
			tensor2VirtLabels=[]
			tensor1PhysLabels=[mpo1.left_label, mpo1.right_label, mpo1.physout_label, mpo1.physin_label]
			tensor2PhysLabels=[mpo2.left_label, mpo2.right_label, mpo2.physout_label, mpo2.physin_label]
		elif i==0 and mpo1.boundaryType=="O":
			tensor1VirtLabels=[mpo1.right_label]
			tensor2VirtLabels=[mpo2.right_label]
			tensor1PhysLabels=[mpo1.left_label, mpo1.physout_label, mpo1.physin_label]
			tensor2PhysLabels=[mpo2.left_label, mpo2.physout_label, mpo2.physin_label]
		elif i==len(mpo1)-1 and mpo1.boundaryType=="O":
			tensor1VirtLabels=[mpo1.left_label]
			tensor2VirtLabels=[mpo2.left_label]
			tensor1PhysLabels=[mpo1.right_label, mpo1.physout_label, mpo1.physin_label]
			tensor2PhysLabels=[mpo2.right_label, mpo2.physout_label, mpo2.physin_label]
		else:
			tensor1VirtLabels=[mpo1.left_label,mpo1.right_label]
			tensor2VirtLabels=[mpo2.left_label,mpo2.right_label]
			tensor1PhysLabels=[mpo1.physout_label, mpo1.physin_label]
			tensor2PhysLabels=[mpo2.physout_label, mpo2.physin_label]
		newTensor = tnc.directSumTensor(tensor1, tensor2, 
						tensor1VirtLabels=tensor1VirtLabels, 
						tensor2VirtLabels=tensor2VirtLabels, 
						tensor1PhysLabels=tensor1PhysLabels, 
						tensor2PhysLabels=tensor2PhysLabels)
		newTensors.append(newTensor)

	newMpo = MatrixProductOperator(newTensors, left_label=mpo1.left_label, right_label=mpo1.right_label, physout_label=mpo1.physout_label, physin_label=mpo1.physin_label)

	return newMpo



def directSum_mps_mps_to_mps(mps1, mps2):
	assert mps1.isKet==mps2.isKet, "directSum_mps_mps_to_mps assumes mps1.isKet==mps2.isKet, but\nmps1.isKet == {0}\nmps2.isKet == {1}\nok?".format(mps1.isKet, mps2.isKet)
	assert mps1.boundaryType==mps2.boundaryType, "directSum_mps_mps_to_mps assumes mps1.boundaryType==mps2.boundaryType, but\nmps1.boundaryType == {0}\nmps2.boundaryType == {1}\nok?".format(mps1.boundaryType, mps2.boundaryType)

	newTensors=[]
	for i,(tensor1,tensor2) in enumerate(zip(mpo1.data,mpo2.data)):
		if i==0 and i==len(mps1)-1 and mpo1.boundaryType=="O":
			tensor1VirtLabels=[]
			tensor2VirtLabels=[]
			tensor1PhysLabels=[mps1.left_label, mps1.right_label, mps1.phys_label]
			tensor2PhysLabels=[mps2.left_label, mps2.right_label, mps2.phys_label]
		elif i==0 and mpo1.boundaryType=="O":
			tensor1VirtLabels=[mps1.right_label]
			tensor2VirtLabels=[mps2.right_label]
			tensor1PhysLabels=[mps1.left_label, mps1.phys_label]
			tensor2PhysLabels=[mps2.left_label, mps2.phys_label]
		elif i==len(mps1)-1 and mpo1.boundaryType=="O":
			tensor1VirtLabels=[mps1.left_label]
			tensor2VirtLabels=[mps2.left_label]
			tensor1PhysLabels=[mps1.right_label, mps1.phys_label]
			tensor2PhysLabels=[mps2.right_label, mps2.phys_label]
		else:
			tensor1VirtLabels=[mps1.left_label,mps1.right_label]
			tensor2VirtLabels=[mps2.left_label,mps2.right_label]
			tensor1PhysLabels=[mps1.phys_label]
			tensor2PhysLabels=[mps2.phys_label]
		newTensor = tnc.directSumTensor(tensor1, tensor2, 
						tensor1VirtLabels=tensor1VirtLabels, 
						tensor2VirtLabels=tensor2VirtLabels, 
						tensor1PhysLabels=tensor1PhysLabels, 
						tensor2PhysLabels=tensor2PhysLabels)
		newTensors.append(newTensor)

	newMps = MatrixProductState(newTensors, left_label=mps1.left_label, right_label=mps1.right_label, phys_label=mps1.phys_label)

	return newMps


#def concatenate_mps_mps
#def concatenate_mpo_mpo