from __future__ import (absolute_import, division,
						print_function, unicode_literals)


from tncontract import tensor_instant as tni
from tncontract.onedim import onedim_core as odc



def random_mps(length, bondDims, physDims, left_label="ket_left", right_label="ket_right", phys_label="ket_phys", boundaryType="O", dtype=complex):
	if isinstance(bondDims,int):
		if boundaryType=="O":
			bondDims = [1]+[bondDims]*(length-1)+[1]
		elif boundaryType=="P" or boundaryType=="S":
			bondDims = [bondDims]*(length+1)
		else:
			raise ValueError("boundaryType must be O, P, or S")
	elif len(bondDims)==length+1:
		bondDims = bondDims
	elif len(bondDims)==length:
		bondDims = bondDims+[bondDims[0]]
	elif len(bondDims)==length-1:
		bondDims = [1]+bondDims+[1]
	else:
		raise ValueError("bondDims and length are not matching")

	if isinstance(physDims,int):
		physDims = [physDims]*length
	elif len(physDims)==length:
		physDims = physDims
	else:
		raise ValueError("physDims and length are not matching")

	randomTensors=[]
	for site in range(length):
		randomTensors.append(tni.random_tensor((bondDims[site],bondDims[site+1],physDims[site]),labels=(left_label,right_label,phys_label), dtype=dtype))
	randomMps = odc.MatrixProductState(tensors=randomTensors, left_label=left_label, right_label=right_label, phys_label=phys_label, boundaryType=boundaryType)
	randomMps.normalise()
	return randomMps



def random_mpo(length, bondDims, physDims, left_label="mpo_left", right_label="mpo_right", physout_label="mpo_physout", physin_label="mpo_physin", offset=0, boundaryType="O", dtype=complex, forceHermite=False):
	if isinstance(bondDims,int):
		if boundaryType=="O":
			bondDims = [1]+[bondDims]*(length-1)+[1]
		elif boundaryType=="P" or boundaryType=="S":
			bondDims = [bondDims]*(length+1)
		else:
			raise ValueError("boundaryType must be O, P, or S")
	elif len(bondDims)==length+1:
		bondDims = bondDims
	elif len(bondDims)==length:
		bondDims = bondDims+[bondDims[0]]
	elif len(bondDims)==length-1:
		bondDims = [1]+bondDims+[1]
	else:
		raise ValueError("bondDims and length are not matching")

	if isinstance(physDims,int):
		physDims = [physDims]*length
	elif len(physDims)==length:
		physDims = physDims
	else:
		raise ValueError("physDims and length are not matching")

	randomTensors=[]
	for site in range(length):
		randomTensors.append(tni.random_tensor((bondDims[site],bondDims[site+1],physDims[site],physDims[site]),labels=(left_label,right_label,physout_label,physin_label),dtype=dtype))
	randomMpo = odc.MatrixProductOperator(tensors=randomTensors, left_label=left_label, right_label=right_label, physout_label=physout_label, physin_label=physin_label, offset=offset, boundaryType=boundaryType)
	if forceHermite:
		randomMpo.forceHermite()
	return randomMpo


def zeros_mpo(length, bondDims, physDims, left_label="mpo_left", right_label="mpo_right", physout_label="mpo_physout", physin_label="mpo_physin", offset=0, boundaryType="O", dtype=complex):
	if isinstance(bondDims,int):
		if boundaryType=="O":
			bondDims = [1]+[bondDims]*(length-1)+[1]
		elif boundaryType=="P" or boundaryType=="S":
			bondDims = [bondDims]*(length+1)
		else:
			raise ValueError("boundaryType must be O, P, or S")
	elif len(bondDims)==length+1:
		bondDims = bondDims
	elif len(bondDims)==length:
		bondDims = bondDims+[bondDims[0]]
	elif len(bondDims)==length-1:
		bondDims = [1]+bondDims+[1]
	else:
		raise ValueError("bondDims and length are not matching")

	if isinstance(physDims,int):
		physDims = [physDims]*length
	elif len(physDims)==length:
		physDims = physDims
	else:
		raise ValueError("physDims and length are not matching")

	zerosTensors=[]
	for site in range(length):
		zerosTensors.append(tni.zeros_tensor((bondDims[site],bondDims[site+1],physDims[site],physDims[site]),labels=(left_label,right_label,physout_label,physin_label),dtype=dtype))
	zerosMpo = odc.MatrixProductOperator(tensors=zerosTensors, left_label=left_label, right_label=right_label, physout_label=physout_label, physin_label=physin_label, offset=offset, boundaryType=boundaryType)
	return zerosMpo


def identity_mpo(length, physDims, left_label="mpo_left", right_label="mpo_right", physout_label="mpo_physout", physin_label="mpo_physin", offset=0, boundaryType="O", dtype=complex):
	if isinstance(physDims,int):
		physDims = [physDims]*length

	identityTensors=[]
	for site in range(length):
		a = tni.identity_tensor(physDims[site], physoutLabel=physout_label, physinLabel=physin_label, virtLabels=(left_label,right_label), dtype=dtype)
		identityTensors.append(a)
		#identityTensors.append(tni.identity_tensor(physDims[site], physoutLabel=physout_label, physinLabel=physin_label, virtLabels=(left_label,right_label), dtype=dtype))
	identityMpo = odc.MatrixProductOperator(tensors=identityTensors, left_label=left_label, right_label=right_label, physout_label=physout_label, physin_label=physin_label, offset=offset, boundaryType=boundaryType)
	return identityMpo



def random_mps_like(mps, bondDims=None, physDims=None, left_label=None, right_label=None, phys_label=None):
	if bondDims is None:
		bondDims = mps.bondDims()
	if physDims is None:
		physDims = mps.physDims()
	if left_label is None:
		left_label = mps.left_label
	if right_label is None:
		right_label = mps.right_label
	if phys_label is None:
		if isinstance(mps, odc.MatrixProductState):
			phys_label = mps.phys_label
		else:
			phys_label = "mps_phys_byRML"

	return random_mps(len(mps), bondDims, physDims, label=left_label, right_label=right_label, phys_label=phys_label, boundaryType=mps.boundaryType, dtype=mps[0].dtype)


def random_mpo_like(mpo, forceHermite=False):
	return random_mpo(len(mpo), mpo.bondDims(), mpo.physDims(), left_label=mpo.left_label, right_label=mpo.right_label, physout_label=mpo.physout_label, physin_label=mpo.physin_label, offset=mpo.offset, boundaryType=mpo.boundaryType, dtype=mpo[0].dtype, forceHermite=forceHermite)


def zeros_mpo_like(mpo):
	return zeros_mpo(len(mpo), mpo.bondDims(), mpo.physDims(), left_label=mpo.left_label, right_label=mpo.right_label, physout_label=mpo.physout_label, physin_label=mpo.physin_label, offset=mpo.offset, boundaryType=mpo.boundaryType, dtype=mpo[0].dtype)

def identity_mpo_like(mpo):
	return identity_mpo(len(mpo), mpo.physDims(), left_label=mpo.left_label, right_label=mpo.right_label, physout_label=mpo.physout_label, physin_label=mpo.physin_label, offset=mpo.offset, boundaryType=mpo.boundaryType, dtype=mpo[0].dtype)
