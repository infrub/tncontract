from __future__ import (absolute_import, division,
						print_function, unicode_literals)

"""
onedim_core
==========

Core module for onedimensional tensor networks
"""

from tncontract import tensor_core as tnc
from tncontract import label as lbl
from tncontract.onedim import onedim_core as odc
from tncontract.onedim import onedim_instant as odi
from tncontract.onedim import onedim_aligner as oda
from math import sqrt



def pile_mps_mps_to_mps(mps1, mps2, new_left_label=None, new_right_label=None, new_phys_label=None):
	assert mps1.isKet==mps2.isKet, "mps1.isKet != mps2.isKet"

	if new_left_label is None: new_left_label=mps1.left_label
	if new_right_label is None: new_right_label=mps1.right_label
	if new_phys_label is None: new_phys_label=mps1.phys_label

	mps1 = mps1.copy()
	mps2 = mps2.copy()
	mps1.lbl.unique_labels()
	mps2.lbl.unique_labels()
	new_tensors = []
	for i in range(len(mps2)):
		new_tensor = tnc.contract(mps1[i], mps2[i], [], [])
		new_tensor.fuse_indices([mps1.left_label, mps2.left_label], new_left_label)
		new_tensor.fuse_indices([mps1.right_label, mps2.right_label], new_right_label)
		new_tensor.fuse_indices([mps1.phys_label, mps2.phys_label], new_phys_label)
		new_tensors.append(new_tensor)
	new_mps = odc.MatrixProductState(new_tensors, new_left_label, new_right_label, new_phys_label, isKet=mps1.isKet)
	return new_mps

def pile_mpo_mpo_to_mpo(mpo1, mpo2, new_left_label=None, new_right_label=None, new_physout_label=None, new_physin_label=None):
	if new_left_label is None: new_left_label=mpo1.left_label
	if new_right_label is None: new_right_label=mpo1.right_label
	if new_physout_label is None: new_physout_label=mpo1.physout_label
	if new_physin_label is None: new_physin_label=mpo1.physin_label
	mpo1 = mpo1.copy()
	mpo2 = mpo2.copy()
	mpo1.unique_labels()
	mpo2.unique_labels()
	new_tensors = []
	for i in range(len(mpo2)):
		new_tensor = tnc.contract(mpo1[i], mpo2[i], [], [])
		new_tensor.fuse_indices([mpo1.left_label, mpo2.left_label], new_left_label)
		new_tensor.fuse_indices([mpo1.right_label, mpo2.right_label], new_right_label)
		new_tensor.fuse_indices([mpo1.physout_label, mpo2.physout_label], new_physout_label)
		new_tensor.fuse_indices([mpo1.physin_label, mpo2.physin_label], new_physin_label)
		new_tensors.append(new_tensor)
	new_mpo = odc.MatrixProductOperator(new_tensors, new_left_label, new_right_label, new_physout_label, new_physin_label)
	return new_mpo


def pile_mpsKet_mpsBra_to_mpo(mpsKet, mpsBra, new_left_label="left_bypMMKMB", new_right_label="right_bypMMKMB", new_physout_label="physout_bypMMKMB", new_physin_label="physin_bypMMKMB"):
	assert not(mpsBra.isKet), "mpsBra is assumed to be a bra, but this is a ket."
	assert mpsKet.isKet, "mpsKet is assumed to be a ket, but this is a bra."

	mpsBra = mpsBra.copy()
	mpsKet = mpsKet.copy()
	mpsBra.unique_labels()
	mpsKet.unique_labels()
	new_tensors = []
	for i in range(len(mpsKet)):
		new_tensor = tnc.contract(mpsBra[i], mpsKet[i], [], [])
		new_tensor.fuse_indices([mpsBra.left_label, mpsKet.left_label], new_left_label)
		new_tensor.fuse_indices([mpsBra.right_label, mpsKet.right_label], new_right_label)
		new_tensor.replace_label([mpsKet.phys_label, mpsBra.phys_label], [new_physout_label, new_physin_label])
		new_tensors.append(new_tensor)
	new_mpo = odc.MatrixProductOperator(new_tensors, new_left_label, new_right_label, new_physout_label, new_physin_label)
	return new_mpo



def snap_mpo_component_to_mps_component(mpoTensor, physout_label, physin_label, new_phys_label):
	mpsTensor = mpoTensor.copy()
	mpsTensor.fuse_indices([physout_label,physin_label],new_phys_label)
	return mpsTensor

def unsnap_mps_component_to_mpo_component(oldTensor, phys_label, new_physout_label, new_physin_label):
	oldTensorPhysDim = oldTensor.index_dimension(phys_label)
	newTensorPhysDim = int(sqrt(oldTensorPhysDim))
	if oldTensorPhysDim != newTensorPhysDim**2:
		raise(Exception("For unsnap mps, physdim must be square number! got "+str(oldTensorPhysDim)))

	newTensor = oldTensor.copy()
	newTensor.split_index(phys_label, [newTensorPhysDim,newTensorPhysDim], [new_physout_label,new_physin_label])
	return newTensor


def snap_mpo_to_mps(mpo, new_phys_label="phys_bySMM"):
	physout_label = mpo.physout_label
	physin_label = mpo.physin_label
	left_label = mpo.left_label
	right_label = mpo.right_label

	newTensors=[]

	for oldTensor in mpo.data:
		newTensor = snap_mpo_component_to_mps_component(oldTensor, physout_label, physin_label, new_phys_label)
		newTensors.append(newTensor)

	mps = odc.MatrixProductState(newTensors, left_label=left_label, right_label=right_label, phys_label=new_phys_label)
	return mps

def unsnap_mps_to_mpo(mps, new_physout_label="physout_byUSMM", new_physin_label="physin_byUSMM"):
	phys_label = mps.phys_label
	left_label = mps.left_label
	right_label = mps.right_label

	newTensors=[]

	for oldTensor in mps.data:
		newTensor = unsnap_mps_component_to_mpo_component(oldTensor, phys_label, new_physout_label, new_physin_label)
		newTensors.append(newTensor)

	mpo = odc.MatrixProductOperator(newTensors, left_label=left_label, right_label=right_label, physout_label=new_physout_label, physin_label=new_physin_label)
	return mpo







def snap_superMpo_to_mpo(mpoBra, mpoKet):
	mpoKet.transpose()
	re = pile_mpo_mpo_to_mpo(mpoBra, mpoKet)
	mpoKet.transpose()
	return re

def state_to_dm(mpsKet):
	mpsBra = mpsKet.adjointed()
	dmMpo = pile_mpsKet_mpsBra_to_mpo(mpsKet, mpsBra)
	return dmMpo






def mpo_to_snapped_neumann_superMpo(H):
	I = odi.identity_mpo_like(H)
	HEmpI = snap_superMpo_to_mpo(H, I)
	IEmpH = snap_superMpo_to_mpo(I, H)

	HEmpI *= -1j
	IEmpH *= 1j

	temp = HEmpI + IEmpH
	return temp

def mpo_to_snapped_neumann_aligned_superMpo(length, miniH):
	return mpo_to_snapped_neumann_superMpo(oda.align_mpo_to_mpo(length, miniH))

def matrixss_to_snapped_neumann_aligned_superMpo(length, mss):
	return mpo_to_snapped_neumann_superMpo(oda.align_matrixss_to_mpo(length, mss))






def mpo_to_snapped_disspation_superMpo(V):
	adjV = V.adjointed()
	I = odi.identity_mpo_like(V)

	VEmpAdjV = snap_superMpo_to_mpo(V, adjV)

	adjVV = adjV * V

	adjVVEmpI = snap_superMpo_to_mpo(adjVV, I)
	IEmpAdjVV = snap_superMpo_to_mpo(I, adjVV)
	
	adjVVEmpI *= -0.5
	IEmpAdjVV *= -0.5
	
	temp = VEmpAdjV + (adjVVEmpI + IEmpAdjVV)

	return temp

def mpo_to_aligned_snapped_disspation_superMpo(length,V): #miniV
	return oda.align_mpo_to_mpo(length, mpo_to_snapped_disspation_superMpo(V))

def matrixs_to_aligned_snapped_disspation_superMpo(length,ms):
	return mpo_to_aligned_snapped_disspation_superMpo(length,oda.matrixs_to_mpo(ms))

def matrixss_to_aligned_snapped_disspation_superMpo(length,mss):
	re = matrixs_to_aligned_snapped_disspation_superMpo(length, mss[0])
	for ms in mss[1:]:
		temp = matrixs_to_aligned_snapped_disspation_superMpo(length, ms)
		re = re + temp
	return re




def matrixss_matrixss_to_snapped_aligned_lindbladian_superMpo(length, hMss, kMss):
	h = matrixss_to_snapped_neumann_aligned_superMpo(length, hMss)
	k = matrixss_to_aligned_snapped_disspation_superMpo(length, kMss)
	return h + k