from __future__ import (absolute_import, division,
						print_function, unicode_literals)

"""
onedim_core
==========

Core module for onedimensional tensor networks
"""

from tncontract import tensor_core as tnc
from tncontract.onedim import onedim_core as odc
from tncontract import tensor_instant as tni
from tncontract import matrices as mts



def tensor_to_unison_mpo(length, tensor, boundaryType="O", left_label="left", right_label="right", physout_label="physout", physin_label="physin"):
	if boundaryType=="O":
		tensors = [tensor.copy() for _ in range(length)]
		tensors[0].data = tensors[0].data[:1,:,:,:]
		tensors[-1].data = tensors[-1].data[:,-1:,:,:]
	else:
		tensors = [tensor.copy() for _ in range(length)]

	return odc.MatrixProductOperator(tensors, left_label=left_label, right_label=right_label, physout_label=physout_label, physin_label=physin_label, boundaryType=boundaryType)


def align_mpo_to_mpo(length, miniMpo, boundaryType="O"):
	if boundaryType=="P":
		return NotImplemented
	else:
		oldMpo = miniMpo
		oldBondDims = oldMpo.bondDims()
		if oldBondDims[0] != 1:
			raise Exception("fsadfasfs")
		if oldBondDims[-1] != 1:
			raise Exception("fasfasasd")
		oldBondIvss = [list(range(oldBondDim)) for oldBondDim in oldBondDims]
		W = sum(oldBondDims)
		unusedNewBondIvs=list(range(0,W))
		newBondIvss = [[unusedNewBondIvs.pop(0) for oldBondIv in oldBondIvs] for oldBondIvs in oldBondIvss]
		AAA = 0
		ZZZ = W-1

		P = miniMpo.physDim(0)
		IMat = mts.identity(P)

		for tensor in miniMpo:
			tensor.move_indices_to_top([miniMpo.left_label, miniMpo.right_label, miniMpo.physout_label, miniMpo.physin_label])
		miniTensors = miniMpo.data

		bigTensor = tni.zeros_tensor((W,W,P,P),[miniMpo.left_label, miniMpo.right_label, miniMpo.physout_label, miniMpo.physin_label])

		bigTensor.data[AAA, AAA]=IMat
		bigTensor.data[ZZZ, ZZZ]=IMat

		for miniSite,miniTensor in enumerate(miniTensors):
			leftOldBondIvs = oldBondIvss[miniSite]
			rightOldBondIvs = oldBondIvss[miniSite+1]
			leftNewBondIvs = newBondIvss[miniSite]
			rightNewBondIvs = newBondIvss[miniSite+1]
			

			for leftOldBondIv, leftNewBondIv in zip(leftOldBondIvs, leftNewBondIvs):
				for rightOldBondIv, rightNewBondIv in zip(rightOldBondIvs, rightNewBondIvs):
					bigTensor.data[leftNewBondIv,rightNewBondIv] = miniTensor.data[leftOldBondIv,rightOldBondIv]


	return tensor_to_unison_mpo(length, bigTensor, boundaryType=boundaryType, left_label=miniMpo.left_label, right_label=miniMpo.right_label, physout_label=miniMpo.physout_label, physin_label=miniMpo.physin_label)



def matrixs_to_mpo(ms,left_label="left",right_label="right",physout_label="physout",physin_label="physin"):
	tensors = []
	#print("mpo_Matrixs")
	#print("ms", ms)
	for m in ms:
		tensor = tnc.Tensor(m, [physout_label,physin_label])
		tensor.add_dummy_index(left_label)
		tensor.add_dummy_index(right_label)
		tensors.append(tensor)
		#print("tensor",tensor)
	mpo = odc.MatrixProductOperator(tensors, left_label=left_label, right_label=right_label, physout_label=physout_label, physin_label=physin_label)
	#print("mpo_Matrixs returned")
	#print(mpo)
	#print(mpo.data)
	return mpo


def align_matrixs_to_mpo(length, ms, boundaryType="O", left_label="left",right_label="right",physout_label="physout",physin_label="physin"):
	#print("ms",ms)
	miniMpo = matrixs_to_mpo(ms, left_label=left_label, right_label=right_label, physout_label=physout_label, physin_label=physin_label)
	lineMpo = align_mpo_to_mpo(length, miniMpo, boundaryType=boundaryType)
	return lineMpo


def align_matrixss_to_mpo(length, mss, boundaryType="O", left_label="left",right_label="right",physout_label="physout",physin_label="physin"):
	ms = mss[0]
	wholeLineMpo = align_matrixs_to_mpo(length, ms, boundaryType=boundaryType, left_label=left_label, right_label=right_label, physout_label=physout_label, physin_label=physin_label)
	for ms in mss[1:]:
		lineMpo = align_matrixs_to_mpo(length, ms, boundaryType=boundaryType, left_label=left_label, right_label=right_label, physout_label=physout_label, physin_label=physin_label)
		wholeLineMpo = wholeLineMpo + lineMpo
	return wholeLineMpo