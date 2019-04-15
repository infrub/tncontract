import sys
sys.path.append('../')

from tncontract import *
from tncontract.onedim import *
from math import sqrt





class HamiltonianCanonisingVariationalOptimizer:
	def __init__(self, hamiltonianMpo, initStateMpsKet=None, mpsBondDims=None, which="SA", printer=(lambda *x: 0)):
		self.op_left_label = "op_left"
		self.op_right_label = "op_right"
		self.op_physout_label = "op_physout"
		self.op_physin_label = "op_physin"
		self.ket_left_label = "ket_left"
		self.ket_right_label = "ket_right"
		self.ket_phys_label = "ket_phys"
		self.bra_left_label = "bra_left"
		self.bra_right_label = "bra_right"
		self.bra_phys_label = "bra_phys"

		self.boundaryType = hamiltonianMpo.boundaryType
		if self.boundaryType=="P":
			raise Exception("HamiltonianCanonisingVariationalOptimizer can't varopt periodic")
		self.length=len(hamiltonianMpo)
		self.which=which
		if which=="LA": self.bestL = float("-inf")
		if which=="SA": self.bestL = float("inf")
		if which=="LM": self.bestL = float(0)
		if which=="SM": self.bestL = float("inf")
		self.printer = printer

		hamiltonianMpo.replace_left_label(self.op_left_label)
		hamiltonianMpo.replace_right_label(self.op_right_label)
		hamiltonianMpo.replace_physout_label(self.op_physout_label)
		hamiltonianMpo.replace_physin_label(self.op_physin_label)
		self.hamiltonianMpo=hamiltonianMpo

		if initStateMpsKet is None:
			if mpsBondDims is None:
				raise Exception("You need one of initStateMpsKet or mpsBondDims")
			else:
				initMpsPhysDims = [hamiltonianMpo.physinDim(site) for site in range(len(hamiltonianMpo))]
				mpsKet = random_mps(length=len(hamiltonianMpo), bondDims=mpsBondDims, physDims=initMpsPhysDims, left_label=self.ket_left_label, right_label=self.ket_right_label, phys_label=self.ket_phys_label, boundaryType=self.boundaryType, dtype=complex)
				self.stateMpsKet = mpsKet

		else:
			self.stateMpsKet=initStateMpsKet

		self.leftSOSmemo={}
		self.rightSOSmemo={}

	def __len__(self):
		return self.length

	def getOneKet(self,site):
		if site=="NOSITE":
			dim=self.stateMpsKet[0].index_dimension(self.ket_left_label)
			return identity_tensor(dim, self.ket_left_label, self.ket_right_label)
		return self.stateMpsKet[site]

	def setOneKet(self,site,tensor):
		self.stateMpsKet[site] = tensor
		self.careAfterMpsOneSiteChanged(site)

	def getOneBra(self,site):
		if site=="NOSITE":
			dim=self.stateMpsKet[0].index_dimension(self.ket_left_label)
			return identity_tensor(dim, self.bra_left_label, self.bra_right_label)
		oneKet = self.getOneKet(site)
		oneBra = oneKet.conjugated()
		oneBra.replace_label([self.ket_left_label,self.ket_right_label,self.ket_phys_label],[self.bra_left_label,self.bra_right_label,self.bra_phys_label])
		return oneBra

	def getOneOp(self,site):
		if site=="NOSITE":
			dim=self.hamiltonianMpo[0].index_dimension(self.op_left_label)
			return identity_tensor(dim, self.op_left_label, self.op_right_label)
		return self.hamiltonianMpo[site]


	def getLeftSOS(self,pun): #O(2pWD^3+ppWWD^2)
		if pun<0 or len(self)<pun:
			raise Exception("leftSOSmemo can have only [0] to [len]")
		if pun in self.leftSOSmemo:
			return self.leftSOSmemo[pun]
		if pun==0:
			oneKet = self.getOneKet("NOSITE")
			oneOp = self.getOneOp("NOSITE")
			oneBra = self.getOneBra("NOSITE")
			temp = contract(oneKet,oneOp,[],[])
			newSOS = contract(temp,oneBra,[],[])
		else:
			predSOS = self.getLeftSOS(pun-1)
			oneKet = self.getOneKet(pun-1)
			oneOp = self.getOneOp(pun-1)
			oneBra = self.getOneBra(pun-1)
			temp1 = contract(predSOS,oneKet,self.ket_right_label,self.ket_left_label)
			temp2 = contract(temp1,oneOp,[self.op_right_label,self.ket_phys_label],[self.op_left_label,self.op_physin_label])
			newSOS = contract(temp2,oneBra,[self.bra_right_label,self.op_physout_label],[self.bra_left_label,self.bra_phys_label])
		self.leftSOSmemo[pun]=newSOS
		return newSOS

	def getRightSOS(self,pun): #O(2pWD^3+ppWWD^2)
		if pun<0 or len(self)<pun:
			raise Exception("leftSOSmemo can have only [0] to [len]")
		if pun in self.rightSOSmemo:
			return self.rightSOSmemo[pun]
		if pun==len(self):
			oneKet = self.getOneKet("NOSITE")
			oneOp = self.getOneOp("NOSITE")
			oneBra = self.getOneBra("NOSITE")
			temp = contract(oneKet,oneOp,[],[])
			newSOS = contract(temp,oneBra,[],[])
		else:
			predSOS = self.getRightSOS(pun+1)
			oneKet = self.getOneKet(pun)
			oneOp = self.getOneOp(pun)
			oneBra = self.getOneBra(pun)
			temp1 = contract(predSOS,oneKet,self.ket_left_label,self.ket_right_label)
			temp2 = contract(temp1,oneOp,[self.op_left_label,self.ket_phys_label],[self.op_right_label,self.op_physin_label])
			newSOS = contract(temp2,oneBra,[self.bra_left_label,self.op_physout_label],[self.bra_right_label,self.bra_phys_label])
		self.rightSOSmemo[pun]=newSOS
		return newSOS

	def careAfterMpsOneSiteChanged(self,site):
		for i in range(site+1,len(self)+1):
			self.leftSOSmemo.pop(i,None)
		for i in range(0,site+1):
			self.rightSOSmemo.pop(i,None)




	def oneSiteVarOpt(self,focusingSite,algorithmName="auto",**kwargs):
		self.printer(1,"\n")
		if algorithmName=="auto":
			"""
			W = int(sqrt(self.hamiltonianMpo.leftDim(focusingSite)*self.hamiltonianMpo.rightDim(focusingSite)))
			D = int(sqrt(self.stateMpsKet.leftDim(focusingSite)*self.stateMpsKet.rightDim(focusingSite)))
			if 2*W<D:
				algorithmName = "TenEigh"
			else:
				algorithmName = "MatEigh"
			"""
			algorithmName = "MatEigh"
			self.printer(5,"choosed algorithmName= ",algorithmName)

		self.printer(1, "optimizing site=", focusingSite)
		self.stateMpsKet.left_canonise_up_to(focusingSite)
		self.stateMpsKet.right_canonise_up_to(focusingSite+1)

		leftSOSTen=self.getLeftSOS(focusingSite)
		rightSOSTen=self.getRightSOS(focusingSite+1)
		midOTen=self.getOneOp(focusingSite)

		focusingKet = self.getOneKet(focusingSite)
		focusingSitePhysDim=focusingKet.index_dimension(self.ket_phys_label)
		focusingSiteLeftDim=focusingKet.index_dimension(self.ket_left_label)
		focusingSiteRightDim=focusingKet.index_dimension(self.ket_right_label)

		ketShape = (focusingSiteLeftDim, focusingSitePhysDim, focusingSiteRightDim)
		#vectorDim = focusingSiteLeftDim * focusingSitePhysDim * focusingSiteRightDim
		#dtype = focusingKet.dtype

		if algorithmName in ("MatEigh", "MatEigsh", "MatPower", "TenEigsh"):
			def ketVec_KetTen(ket):
				vector = tensor_to_vector(ket, [self.ket_left_label,self.ket_phys_label,self.ket_right_label])
				return vector

			def ketTen_KetVec(vector):
				ket = vector_to_tensor(vector, ketShape, [self.ket_left_label,self.ket_phys_label,self.ket_right_label])
				return ket

		if algorithmName == "TenPower":
			def sca_ConjKetTen_KetTen(conjKetTen, ketTen):
				temp = contract(conjKetTen, ketTen, [self.ket_left_label,self.ket_phys_label,self.ket_right_label], [self.ket_left_label,self.ket_phys_label,self.ket_right_label])
				return temp.data.real


		if algorithmName in ("MatEigh", "MatEigsh", "MatPower"):
			temp = contract(leftSOSTen,midOTen,[self.op_right_label],[self.op_left_label])
			wholeSOSTen = contract(temp, rightSOSTen, [self.op_right_label, self.ket_left_label, self.op_left_label, self.bra_left_label], [self.op_left_label, self.ket_right_label, self.op_right_label, self.bra_right_label])

			wholeH = wholeSOSTen.to_matrix(row_labels=[self.bra_right_label,self.op_physout_label,self.bra_left_label], column_labels=[self.ket_right_label,self.op_physin_label,self.ket_left_label])

		if algorithmName in ("TenEigsh", "TenPower"):
			def wholeSOSKet_Ket(ket):
				temp1 = contract(ket, leftSOSTen, [self.ket_left_label], [self.ket_right_label])
				temp2 = contract(temp1, midOTen, [self.op_right_label, self.ket_phys_label], [self.op_left_label, self.op_physin_label])
				temp3 = contract(temp2, rightSOSTen, [self.op_right_label, self.ket_right_label,  self.ket_left_label, self.op_left_label, self.bra_left_label], [self.op_left_label, self.ket_left_label, self.ket_right_label, self.op_right_label, self.bra_right_label])
				temp3.replace_label([self.bra_right_label, self.op_physout_label, self.bra_left_label], [self.ket_left_label, self.ket_phys_label, self.ket_right_label])
				return temp3


		if algorithmName == "MatEigh":
			bestL, bestKet = eiglib.getBestEigTen_byMatEigh(wholeH, ketTen_KetVec, ketVec_KetTen, initRatio=self.bestL, initKetTen=focusingKet, which=self.which, printer=self.printer)	#O(2*p^3*D^6*rep+4*p^3*D^6)

		elif algorithmName == "MatEigsh":
			bestL, bestKet = eiglib.getBestEigTen_byMatEigsh(wholeH, ketTen_KetVec, ketVec_KetTen, initRatio=self.bestL, initKetTen=focusingKet, which=self.which, relativeTolerance=kwargs.pop("relativeTolerance", 1e-6), printer=self.printer)

		elif algorithmName == "MatPower":
			bestL, bestKet = eiglib.getBestEigTen_byMatPower(wholeH, ketTen_KetVec, ketVec_KetTen, initRatio=self.bestL, initKetTen=focusingKet, which=self.which, maxRepeatTurnl=kwargs.pop("maxRepeatTurnl", 30), relativeTolerance=kwargs.pop("relativeTolerance", 1e-6), printer=self.printer)

		elif algorithmName == "TenEigsh":
			bestL, bestKet = eiglib.getBestEigTen_byTenEigsh(wholeSOSKet_Ket, ketTen_KetVec, ketVec_KetTen, initRatio=self.bestL, initKetTen=focusingKet, which=self.which, relativeTolerance=kwargs.pop("relativeTolerance", 1e-6), printer=self.printer)

		elif algorithmName == "TenPower":
			bestL, bestKet = eiglib.getBestEigTen_byTenPower(wholeSOSKet_Ket, sca_ConjKetTen_KetTen, initRatio=self.bestL, initKetTen=focusingKet, which=self.which, maxRepeatTurnl=kwargs.pop("maxRepeatTurnl", 30), relativeTolerance=kwargs.pop("relativeTolerance", 1e-6), printer=self.printer)

		else:
			return ValueError("No such algorithmName", algorithmName)


		self.bestL = bestL
		self.setOneKet(focusingSite, bestKet)

		self.printer(1,"optimized.")
		self.printer(1,"now bestL=",self.bestL)

		return self.bestL




	def oneSweepVarOpt(self,algorithmName="auto",**kwargs):
		for focusingSite in range(len(self)-1):
			bestL = self.oneSiteVarOpt(focusingSite, algorithmName=algorithmName, **kwargs)
		for focusingSite in range(len(self)-1,0,-1):
			bestL = self.oneSiteVarOpt(focusingSite, algorithmName=algorithmName, **kwargs)
		return bestL


	def varOpt(self, maxSweepTurnl=30, absoluteTolerance=None, relativeTolerance=None, algorithmName="auto", **kwargs):
		for sweepTurni in range(maxSweepTurnl):
			self.printer(1, "\n\nsweepTurni=", sweepTurni)
			previousBestL=self.bestL
			bestL = self.oneSweepVarOpt(algorithmName=algorithmName, **kwargs)
			if absoluteTolerance is not None:
				if self.which=="SA" and bestL<=absoluteTolerance:
					self.printer(1, "\nabsoluteTolerance reached. break.")
					break
				elif self.which=="LA" and bestL>=absoluteTolerance:
					self.printer(1, "\nabsoluteTolerance reached. break.")
					break
				elif self.which=="SM" and abs(bestL)<=absoluteTolerance:
					self.printer(1, "\nabsoluteTolerance reached. break.")
					break
				elif self.which=="LM" and abs(bestL)>=absoluteTolerance:
					self.printer(1, "\nabsoluteTolerance reached. break.")
					break
			if relativeTolerance is not None:
				#print("bestL",bestL)
				#print("previousBestL",previousBestL)
				if abs(bestL-previousBestL)<=abs(relativeTolerance*bestL):
					self.printer(1, "\nrelativeTolerance reached. break.")
					break
			if bestL==previousBestL:
				self.printer(1, "\nno more opt. break.")
				break

		sweepTurnl = sweepTurni+1
		self.printer(1, "\n\nvaropt done.\nsweepTurnl == "+str(sweepTurnl))
		return bestL, sweepTurnl
