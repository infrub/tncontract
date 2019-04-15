import sys
sys.path.append('../')

from tncontract.eksp import xp as xp
from math import sqrt
from tncontract import tensor_instant as tni


def checkNewEigIsBetter(initL, bestL, which):
	if initL is None:
		return True
	if which=="LA":
		return bestL>=initL
	elif which=="SA":
		return bestL<=initL
	elif which=="LM":
		return abs(bestL)>=abs(initL)
	elif which=="SM":
		return abs(bestL)<=abs(initL)

def getFarrestEig_byPower(opKet_Ket, sca_ConjKet_Ket, conjKet_Ket, norm_Ket, initRatio=None, initKet=None, origin=None, maxRepeatTurnl=30, relativeTolerance=1e-6, printer=(lambda *x: 0)):
	if initRatio is None:
		initRatio = 0
	if initKet is None:
		raise ValueError("getFarrestEig_byPower require argument initKet")

	newKet = initKet
	newRatio = initRatio
	if not origin is None:
		newRatio = newRatio - origin

	for repeatTurni in range(maxRepeatTurnl):
		oldKet = newKet
		oldRatio = newRatio

		newKet = opKet_Ket(oldKet)
		if not origin is None:
			newKet = newKet - oldKet*origin
		#oldConjKet = oldKet.conjugated()
		oldConjKet = conjKet_Ket(oldKet)
		oldSca = sca_ConjKet_Ket(oldConjKet, oldKet)
		newSca = sca_ConjKet_Ket(oldConjKet, newKet)
		newRatio = newSca/oldSca
		newKet = newKet / norm_Ket(newKet)

		if abs(oldRatio-newRatio)<=abs(newRatio*relativeTolerance):
			printer(11, "getFarrestEig_byPower relativeTolerance reached. break.")
			break

		if oldRatio==newRatio:
			printer(11, "getFarrestEig_byPower no more opt. break.")
			break

	repeatTurnl = repeatTurni + 1
	printer(11, "\n\ngetFarrestEig_byPower done.\nrepeatTurnl == "+str(repeatTurnl))

	if not origin is None:
		newRatio = newRatio + origin

	return newRatio, newKet



def getBestEig_byPower(opKet_Ket, sca_ConjKet_Ket, conjKet_Ket, norm_Ket, initRatio, initKet, randomKet_Ket, which="LM", maxRepeatTurnl=30, relativeTolerance=1e-6, printer=(lambda *x: 0)):
	randomKet = randomKet_Ket(initKet)

	if which=="LM":
		bestRatio, bestKet = getFarrestEig_byPower(opKet_Ket, sca_ConjKet_Ket, conjKet_Ket, norm_Ket, initRatio, initKet, maxRepeatTurnl=maxRepeatTurnl, relativeTolerance=relativeTolerance, printer=printer)

	elif which=="LA":
		largestRatio, largestKet = getFarrestEig_byPower(opKet_Ket, sca_ConjKet_Ket, conjKet_Ket, norm_Ket, initRatio=None, initKet=randomKet, maxRepeatTurnl=int(maxRepeatTurnl/3), relativeTolerance=1e-2, printer=printer)
		if largestRatio>=0:
			bestRatio, bestKet = getFarrestEig_byPower(opKet_Ket, sca_ConjKet_Ket, conjKet_Ket, norm_Ket, initRatio, initKet, maxRepeatTurnl=maxRepeatTurnl, relativeTolerance=relativeTolerance, printer=printer)
		else:
			bestRatio, bestKet = getFarrestEig_byPower(opKet_Ket, sca_ConjKet_Ket, conjKet_Ket, norm_Ket, initRatio, initKet, origin=largestRatio, maxRepeatTurnl=maxRepeatTurnl, relativeTolerance=relativeTolerance, printer=printer)

	elif which=="SA":
		largestRatio, largestKet = getFarrestEig_byPower(opKet_Ket, sca_ConjKet_Ket, conjKet_Ket, norm_Ket, initRatio=None, initKet=randomKet, maxRepeatTurnl=int(maxRepeatTurnl/3), relativeTolerance=1e-2, printer=printer)
		if largestRatio<=0:
			bestRatio, bestKet = getFarrestEig_byPower(opKet_Ket, sca_ConjKet_Ket, conjKet_Ket, norm_Ket, initRatio, initKet, maxRepeatTurnl=maxRepeatTurnl, relativeTolerance=relativeTolerance, printer=printer)
		else:
			bestRatio, bestKet = getFarrestEig_byPower(opKet_Ket, sca_ConjKet_Ket, conjKet_Ket, norm_Ket, initRatio, initKet, origin=largestRatio, maxRepeatTurnl=maxRepeatTurnl, relativeTolerance=relativeTolerance, printer=printer)

	else:
		return NotImplemented

	if not checkNewEigIsBetter(initRatio, bestRatio, which):
		printer(1, "initL is better than bestL.\ninitL={0}, bestL={1}\nso skipped.".format(initRatio, bestRatio))
		return initRatio, initKet

	return bestRatio, bestKet



def getBestEigVec_byMatPower(opMat, initRatio, initKetVec, which="LM", maxRepeatTurnl=30, relativeTolerance=1e-6, printer=(lambda *x: 0)):
	opKet_Ket = lambda ketVec: xp.dot(opMat, ketVec)
	sca_ConjKet_Ket = lambda conjKet, ket: xp.dot(conjKet, ket).real
	conjKet_Ket = xp.conj
	norm_Ket = xp.linalg.norm
	randomKet_Ket = lambda ketVec: xp.random.random(*ketVec.shape)
	return getBestEig_byPower(opKet_Ket, sca_ConjKet_Ket, conjKet_Ket, norm_Ket, initRatio, initKetVec, randomKet_Ket, which=which, maxRepeatTurnl=maxRepeatTurnl, relativeTolerance=relativeTolerance, printer=printer)

def getBestEigTen_byMatPower(opMat, ketTen_KetVec, ketVec_KetTen, initRatio, initKetTen, which="LM", maxRepeatTurnl=30, relativeTolerance=1e-6, printer=(lambda *x: 0)):
	initKetVec = ketVec_KetTen(initKetTen)
	bestRatio, bestKetVec = getBestEigVec_byMatPower(opMat, initRatio, initKetVec, which=which, maxRepeatTurnl=30, relativeTolerance=1e-6, printer=printer)
	bestKetTen = ketTen_KetVec(bestKetVec)
	return bestRatio, bestKetTen

def getBestEigTen_byTenPower(opKetTen_KetTen, sca_ConjKetTen_KetTen, initRatio, initKetTen, which="LM", maxRepeatTurnl=30, relativeTolerance=1e-6, printer=(lambda *x: 0)):
	conjKet_Ket = lambda ketTen: ketTen.conjugated()
	norm_Ket = lambda ketTen: ketTen.norm()
	randomKet_Ket = lambda ketTen: tni.random_tensor_like(ketTen)
	return getBestEig_byPower(opKetTen_KetTen, sca_ConjKetTen_KetTen, conjKet_Ket, norm_Ket, initRatio, initKetTen, randomKet_Ket, which=which, maxRepeatTurnl=maxRepeatTurnl, relativeTolerance=relativeTolerance, printer=printer)




def getBestEigVec_byMatEigsh(opMat, initRatio, initKetVec=None, which="LM", relativeTolerance=1e-6, printer=(lambda *x: 0)):
	if xp.isCupy:
		raise Exception("MatEigsh can work only when xp==scipy")

	dim = opMat.shape[0]
	if dim<=2:
		return getBestEigVec_byMatEigh(opMat, initRatio, initKetVec, which=which, printer=printer)

	Ls, Vs = xp.sparse.linalg.eigsh(opMat, k=1, v0=initKetVec, which=which, tol=relativeTolerance)
	bestRatio = Ls[0]
	bestKetVec = Vs[:,0]

	if not checkNewEigIsBetter(initRatio, bestRatio, which):
		printer(1, "initL is better than bestL.\ninitL={0}, bestL={1}\nso skipped.".format(initRatio, bestRatio))
		return initRatio, initKetVec

	return bestRatio, bestKetVec


def getBestEigTen_byMatEigsh(opMat, ketTen_KetVec, ketVec_KetTen, initRatio=None, initKetTen=None, which="LM", relativeTolerance=1e-6, printer=(lambda *x: 0)):
	if xp.isCupy:
		raise Exception("MatEigsh can work only when xp==scipy")

	if initKetTen is None:
		initKetVec = None
	else:
		initKetVec = ketVec_KetTen(initKetTen)
	bestRatio, bestKetVec = getBestEigVec_byMatEigsh(opMat, initRatio, initKetVec, which, relativeTolerance=relativeTolerance, printer=printer)
	if bestKetVec is None:
		bestKetTen = None
	else:
		bestKetTen = ketTen_KetVec(bestKetVec)
	return bestRatio, bestKetTen


def getBestEigTen_byTenEigsh(opKetTen_KetTen, ketTen_KetVec, ketVec_KetTen, initRatio, initKetTen, which="LM", relativeTolerance=1e-6, printer=(lambda *x: 0)):
	if xp.isCupy:
		raise Exception("TenEigsh can work only when xp==scipy")

	initKetVec = ketVec_KetTen(initKetTen)

	def opKetVec_KetVec(oldKetVec):
		oldKetTen = ketTen_KetVec(oldKetVec)
		newKetTen = opKetTen_KetTen(oldKetTen)
		newKetVec = ketVec_KetTen(newKetTen)
		return newKetVec

	opMat = xp.sparse.linalg.LinearOperator(shape=(initKetVec.shape[0], initKetVec.shape[0]), dtype=initKetVec.dtype, matvec=opKetVec_KetVec)

	bestRatio, bestKetVec = getBestEigVec_byMatEigsh(opMat, initRatio, initKetVec, which, relativeTolerance=relativeTolerance, printer=printer)

	bestKetTen = ketTen_KetVec(bestKetVec)

	return bestRatio, bestKetTen




def matrix_LinearOperator(linearOperator):
	rowDim, colDim = linearOperator.shape
	identityMatrix = xp.identity((colDim))
	matrix = linearOperator.dot(identityMatrix)
	return matrix

def getBestLi_fromAscendingLs(Ls, which):
	if which=="LA":
		bestLi = len(Ls)-1
	elif which=="SA":
		bestLi = 0
	elif which=="LM":
		if abs(Ls[0]) <= abs(Ls[-1]):
			bestLi = len(Ls)-1
		else:
			bestLi = 0
	elif which=="SM":
		lefterIsMinusPun = 0
		righterIsPlusPun = len(Ls)
		while True:
			midPun = int((lefterIsMinusPun+righterIsPlusPun)/2)
			if Ls[midPun-1]<=0:
				lefterIsMinusPun = midPun
			if Ls[midPun]>=0:
				righterIsPlusPun = midPun
			if lefterIsMinusPun == righterIsPlusPun:
				break
		if abs(Ls[midPun-1]) <= abs(Ls[midPun]):
			bestLi = midPun-1
		else:
			bestLi = midPun
	#print("Ls:", Ls)
	#print("bestL:", Ls[bestLi])
	return bestLi

def getBestEigVec_byMatEigh(opMat, initRatio=None, initKetVec=None, which="LM", printer=(lambda *x: 0)):
	Ls, Vs = xp.linalg.eigh(opMat)
	bestLi = getBestLi_fromAscendingLs(Ls, which=which)
	bestRatio = Ls[bestLi]
	bestKetVec = Vs[:,bestLi]

	if not checkNewEigIsBetter(initRatio, bestRatio, which):
		printer(1, "initL is better than bestL.\ninitRatio={0}, bestRatio={1}\nso skipped.".format(initRatio, bestRatio))
		return initRatio, initKetVec

	return bestRatio, bestKetVec


def getBestEigTen_byMatEigh(opMat, ketTen_KetVec, ketVec_KetTen, initRatio=None, initKetTen=None, which="LM", printer=(lambda *x: 0)):
	if xp.isScipy:
		if isinstance(opMat, xp.sparse.linalg.LinearOperator):
			opMat = matrix_LinearOperator(opMat)

	if initKetTen is None:
		initKetVec = None
	else:
		initKetVec = ketVec_KetTen(initKetTen)
	bestRatio, bestKetVec = getBestEigVec_byMatEigh(opMat, initRatio, initKetVec, which, printer=printer)
	if bestKetVec is None:
		bestKetTen = None
	else:
		bestKetTen = ketTen_KetVec(bestKetVec)
	return bestRatio, bestKetTen
