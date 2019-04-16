import sys
sys.path.append('../')

from tncontract import *
from tncontract.onedim import *
from math import sqrt
import logging as _logging
logger = _logging.getLogger("tncontract")





class LiouvillianCanonisingVariationalOptimizer:
	def __init__(self, snappedLMpo, initDmMpo=None, dmBondDims=None, which="SA", boundaryType="O"):
		self.l_left_label = "l_left"
		self.l_right_label = "l_right"
		self.l_physout_label = "l_physout"
		self.l_physin_label = "l_physin"
		self.adjL_left_label = "adjL_left"
		self.adjL_right_label = "adjL_right"
		self.adjL_physout_label = "adjL_physout"
		self.adjL_physin_label = "adjL_physin"
		self.dm_left_label = "dm_left"
		self.dm_right_label = "dm_right"
		self.dm_physout_label = "dm_physout"
		self.dm_physin_label = "dm_physin"
		self.ket_left_label = self.dm_left_label
		self.ket_right_label = self.dm_right_label
		self.ket_phys_label = "ket_phys"
		self.bra_left_label = "bra_left"
		self.bra_right_label = "bra_right"
		self.bra_phys_label = "bra_phys"

		if boundaryType=="P":
			raise Exception("HamiltonianCanonisingVariationalOptimizer can't varopt periodic mps")
		self.boundaryType=boundaryType
		self.length=len(snappedLMpo)
		self.which=which
		if which=="LA": self.bestL = float("-inf")
		if which=="SA": self.bestL = float("inf")
		if which=="LM": self.bestL = float(0)
		if which=="SM": self.bestL = float("inf")

		self.leftSOSmemo={}
		self.rightSOSmemo={}

		snappedLMpo.replace_left_label(self.l_left_label)
		snappedLMpo.replace_right_label(self.l_right_label)
		snappedLMpo.replace_physout_label(self.l_physout_label)
		snappedLMpo.replace_physin_label(self.l_physin_label)
		self.snappedLMpo = snappedLMpo

		snappedAdjLMpo = snappedLMpo.adjointed()
		snappedAdjLMpo.replace_left_label(self.adjL_left_label)
		snappedAdjLMpo.replace_right_label(self.adjL_right_label)
		snappedAdjLMpo.replace_physout_label(self.adjL_physout_label)
		snappedAdjLMpo.replace_physin_label(self.adjL_physin_label)
		self.snappedAdjLMpo = snappedAdjLMpo

		if initDmMpo is None:
			if dmBondDims is None:
				raise Exception("You need one of initDmMpo or dmBondDims")
			else:
				initDmMpoPhysDims = [int(sqrt(snappedLMpo.physinDim(site))) for site in range(len(snappedLMpo))]
				initDmMpo = random_mpo(length=len(snappedLMpo), bondDims=dmBondDims, physDims=initDmMpoPhysDims, left_label=self.dm_left_label, right_label=self.dm_right_label, physout_label=self.dm_physout_label, physin_label=self.dm_physin_label, boundaryType=snappedLMpo.boundaryType, dtype=complex, forceHermite=True)
		else:
			initDmMpo.replace_left_label(self.dm_left_label)
			initDmMpo.replace_right_label(self.dm_right_label)
			initDmMpo.replace_physout_label(self.dm_physout_label)
			initDmMpo.replace_physin_label(self.dm_physin_label)

		self.setDmMpo(initDmMpo)


	def setDmMpo(self, dmMpo):
		self.snappedDmMpsKet = snap_mpo_to_mps(dmMpo, new_phys_label=self.ket_phys_label)
		self.snappedDmMpsKet.replace_left_label(self.ket_left_label)
		self.snappedDmMpsKet.replace_right_label(self.ket_right_label)
		self.careAfterMpsSomeSitesChanged(0, len(self))

	def getDmMpo(self):
		return unsnap_mps_to_mpo(self.snappedDmMpsKet, new_physout_label=self.dm_physout_label, new_physin_label=self.dm_physin_label)

	def setSnappedDmMpsKet_Site(self, site, newMpsComponentTensor):
		self.snappedDmMpsKet[site] = newMpsComponentTensor
		self.careAfterMpsOneSiteChanged(site)


	def __len__(self):
		return self.length

	def getOneKet(self,site):
		if site=="NOSITE":
			dim=self.snappedDmMpsKet[0].index_dimension(self.ket_left_label)
			return identity_tensor(dim,self.ket_left_label,self.ket_right_label)
		return self.snappedDmMpsKet[site]

	def getOneBra(self,site):
		if site=="NOSITE":
			dim=self.snappedDmMpsKet[0].index_dimension(self.ket_left_label)
			return identity_tensor(dim,self.bra_left_label,self.bra_right_label)
		oneKet = self.getOneKet(site)
		oneBra = oneKet.conjugated()
		oneBra.replace_label([self.ket_left_label,self.ket_right_label,self.ket_phys_label],[self.bra_left_label,self.bra_right_label,self.bra_phys_label])
		return oneBra

	def getOneL(self,site):
		if site=="NOSITE":
			dim=self.snappedLMpo[0].index_dimension(self.l_left_label)
			return identity_tensor(dim,self.l_left_label,self.l_right_label)
		return self.snappedLMpo[site]

	def getOneAdjL(self,site):
		if site=="NOSITE":
			dim=self.snappedAdjLMpo[0].index_dimension(self.adjL_left_label)
			return identity_tensor(dim,self.adjL_left_label,self.adjL_right_label)
		return self.snappedAdjLMpo[site]

	def getLeftSOS(self,pun): #O(2pWD^3+ppWWD^2)
		if pun<0 or len(self)<pun:
			raise Exception("leftSOSmemo can have only [0] to [len]")
		if pun in self.leftSOSmemo:
			return self.leftSOSmemo[pun]
		if pun==0:
			oneKet = self.getOneKet("NOSITE")
			oneL = self.getOneL("NOSITE")
			oneAdjL = self.getOneAdjL("NOSITE")
			oneBra = self.getOneBra("NOSITE")
			temp = contract(oneKet,oneL,[],[])
			temp = contract(temp,oneAdjL,[],[])
			newSOS = contract(temp,oneBra,[],[])
		else:
			predSOS = self.getLeftSOS(pun-1)
			oneKet = self.getOneKet(pun-1)
			oneL = self.getOneL(pun-1)
			oneAdjL = self.getOneAdjL(pun-1)
			oneBra = self.getOneBra(pun-1)
			temp = contract(predSOS,oneKet,self.ket_right_label,self.ket_left_label)
			temp = contract(temp,oneL,[self.l_right_label,self.ket_phys_label],[self.l_left_label,self.l_physin_label])
			temp = contract(temp,oneAdjL,[self.adjL_right_label,self.l_physout_label],[self.adjL_left_label,self.adjL_physin_label])
			newSOS = contract(temp,oneBra,[self.bra_right_label,self.adjL_physout_label],[self.bra_left_label,self.bra_phys_label])
		self.leftSOSmemo[pun]=newSOS
		#print("calculated leftSOSmemo["+str(pun)+"]")
		return newSOS

	def getRightSOS(self,pun): #O(2pWD^3+ppWWD^2)
		if pun<0 or len(self)<pun:
			raise Exception("rightSOSmemo can have only [0] to [len]")
		if pun in self.rightSOSmemo:
			return self.rightSOSmemo[pun]
		if pun==len(self):
			oneKet = self.getOneKet("NOSITE")
			oneL = self.getOneL("NOSITE")
			oneAdjL = self.getOneAdjL("NOSITE")
			oneBra = self.getOneBra("NOSITE")
			temp = contract(oneKet,oneL,[],[])
			temp = contract(temp,oneAdjL,[],[])
			newSOS = contract(temp,oneBra,[],[])
		else:
			predSOS = self.getRightSOS(pun+1)
			oneKet = self.getOneKet(pun)
			oneL = self.getOneL(pun)
			oneAdjL = self.getOneAdjL(pun)
			oneBra = self.getOneBra(pun)
			temp = contract(predSOS,oneKet,self.ket_left_label,self.ket_right_label)
			temp = contract(temp,oneL,[self.l_left_label,self.ket_phys_label],[self.l_right_label,self.l_physin_label])
			temp = contract(temp,oneAdjL,[self.adjL_left_label,self.l_physout_label],[self.adjL_right_label,self.adjL_physin_label])
			newSOS = contract(temp,oneBra,[self.bra_left_label,self.adjL_physout_label],[self.bra_right_label,self.bra_phys_label])
		self.rightSOSmemo[pun]=newSOS
		#print("calculated rightSOSmemo["+str(pun)+"]")
		return newSOS



	def careAfterMpsSomeSitesChanged(self,left_pun,right_pun):
		for i in range(left_pun+1,len(self)+1):
			self.leftSOSmemo.pop(i,None)
			#print("popped leftSOSmemo["+str(i)+"]")
		for i in range(0,right_pun):
			self.rightSOSmemo.pop(i,None)
			#print("popped rightSOSmemo["+str(i)+"]")

	def careAfterMpsOneSiteChanged(self,site):
		self.careAfterMpsSomeSitesChanged(site, site+1)



	def oneSiteVarOpt(self,focusingSite,algorithmName,**kwargs):
		logger.log(14, f"varopt site={focusingSite} start.")

		if algorithmName=="auto":
			"""
			if self.boundaryType=="P":
				raise Exception("HamiltonianCanonisingVariationalOptimizer can't varopt periodic mps")
			else:
				W = int(sqrt(self.snappedLMpo.leftdim(focusingSite)*self.snappedLMpo.rightdim(focusingSite)))
				D = int(sqrt(self.snappedDmMpsKet.leftdim(focusingSite)*self.snappedDmMpsKet.rightdim(focusingSite)))
				if 2*W<D:
					algorithmName = "ImplicitLanczos"
				else:
					algorithmName = "ExplicitLanczos"
			"""
			algorithmName = "MatEigh"
			logger.log(12, f"choosed algorithmName={algorithmName}")

		motomoto_left_canonised_up_to = self.snappedDmMpsKet.left_canonise_up_to(focusingSite)
		motomoto_right_canonised_up_to = self.snappedDmMpsKet.right_canonise_up_to(focusingSite+1)
		self.careAfterMpsSomeSitesChanged(motomoto_left_canonised_up_to, motomoto_right_canonised_up_to)

		focusingKet = self.getOneKet(focusingSite)
		focusingSitePhysDim=focusingKet.index_dimension(self.ket_phys_label)
		focusingSiteLeftDim=focusingKet.index_dimension(self.ket_left_label)
		focusingSiteRightDim=focusingKet.index_dimension(self.ket_right_label)

		leftSOSTen=self.getLeftSOS(focusingSite)
		rightSOSTen=self.getRightSOS(focusingSite+1)
		midLTen=self.getOneL(focusingSite)
		midAdjLTen=self.getOneAdjL(focusingSite)

		ketShape = (focusingSiteLeftDim, focusingSitePhysDim, focusingSiteRightDim)
		#vectorDim = focusingSiteLeftDim * focusingSitePhysDim * focusingSiteRightDim
		#dtype = focusingKet.data.dtype


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
			temp = contract(leftSOSTen, midLTen, [self.l_right_label],[self.l_left_label])
			temp = contract(temp, midAdjLTen, [self.adjL_right_label,self.l_physout_label],[self.adjL_left_label,self.adjL_physin_label])
			wholeSOSTen = contract(temp, rightSOSTen, [self.l_right_label, self.adjL_right_label, self.ket_left_label, self.l_left_label, self.adjL_left_label, self.bra_left_label], [self.l_left_label, self.adjL_left_label, self.ket_right_label, self.l_right_label, self.adjL_right_label, self.bra_right_label])

			wholeH = wholeSOSTen.to_matrix(row_labels=[self.bra_right_label,self.adjL_physout_label,self.bra_left_label], column_labels=[self.ket_right_label,self.l_physin_label,self.ket_left_label])

		if algorithmName in ("TenEigsh", "TenPower"):
			def wholeSOSKet_Ket(ket):
				temp = contract(ket, leftSOSTen, [self.ket_left_label], [self.ket_right_label])
				temp = contract(temp, midLTen, [self.l_right_label, self.ket_phys_label], [self.l_left_label, self.l_physin_label])
				temp = contract(temp, midAdjLTen, [self.adjL_right_label, self.l_physout_label], [self.adjL_left_label, self.adjL_physin_label])
				temp = contract(temp, rightSOSTen, [self.ket_right_label, self.l_right_label, self.adjL_right_label, self.ket_left_label, self.l_left_label, self.adjL_left_label, self.bra_left_label], [self.ket_left_label, self.l_left_label, self.adjL_left_label, self.ket_right_label, self.l_right_label, self.adjL_right_label, self.bra_right_label])
				temp.replace_label([self.bra_right_label, self.adjL_physout_label, self.bra_left_label], [self.ket_left_label, self.ket_phys_label, self.ket_right_label])
				return temp


		if algorithmName == "MatEigh":
			bestL, bestKet = eiglib.getBestEigTen_byMatEigh(wholeH, ketTen_KetVec, ketVec_KetTen, initRatio=self.bestL, initKetTen=focusingKet, which=self.which)

		elif algorithmName == "MatEigsh":
			bestL, bestKet = eiglib.getBestEigTen_byMatEigsh(wholeH, ketTen_KetVec, ketVec_KetTen, initRatio=self.bestL, initKetTen=focusingKet, which=self.which, relativeTolerance=kwargs.pop("relativeTolerance", 1e-6))

		elif algorithmName == "MatPower":
			bestL, bestKet = eiglib.getBestEigTen_byMatPower(wholeH, ketTen_KetVec, ketVec_KetTen, initRatio=self.bestL, initKetTen=focusingKet, which=self.which, maxRepeatTurnl=kwargs.pop("maxRepeatTurnl", 30), relativeTolerance=kwargs.pop("relativeTolerance", 1e-6))

		elif algorithmName == "TenEigsh":
			bestL, bestKet = eiglib.getBestEigTen_byTenEigsh(wholeSOSKet_Ket, ketTen_KetVec, ketVec_KetTen, initRatio=self.bestL, initKetTen=focusingKet, which=self.which, relativeTolerance=kwargs.pop("relativeTolerance", 1e-6))

		elif algorithmName == "TenPower":
			bestL, bestKet = eiglib.getBestEigTen_byTenPower(wholeSOSKet_Ket, sca_ConjKetTen_KetTen, initRatio=self.bestL, initKetTen=focusingKet, which=self.which, maxRepeatTurnl=kwargs.pop("maxRepeatTurnl", 30), relativeTolerance=kwargs.pop("relativeTolerance", 1e-6))

		else:
			return ValueError("No such algorithmName", algorithmName)


		self.bestL = bestL
		self.setSnappedDmMpsKet_Site(focusingSite, bestKet)

		logger.log(14, f"varopt site={focusingSite} done. now bestL={self.bestL}")

		return self.bestL




	def oneSweepVarOpt(self,algorithmName="auto",**kwargs):
		for focusingSite in range(len(self)-1):
			bestL = self.oneSiteVarOpt(focusingSite,algorithmName=algorithmName,**kwargs)
		for focusingSite in range(len(self)-1,0,-1):
			bestL = self.oneSiteVarOpt(focusingSite,algorithmName=algorithmName,**kwargs)
		return bestL


	def varOpt(self, maxSweepTurnl=30, absoluteTolerance=None, relativeTolerance=None, algorithmName="auto", **kwargs):
		logger.log(18, f"varopt start.")

		for sweepTurni in range(maxSweepTurnl):
			logger.log(16, f"varopt sweepTurni={sweepTurni} start.")
			previousBestL=self.bestL
			bestL = self.oneSweepVarOpt(algorithmName=algorithmName, **kwargs)
			logger.log(16, f"varopt sweepTurni={sweepTurni} done.")

			if absoluteTolerance is not None:
				if self.which=="SA" and bestL<=absoluteTolerance:
					logger.log(18, "varopt break. absoluteTolerance reached.")
					break
				elif self.which=="LA" and bestL>=absoluteTolerance:
					logger.log(18, "varopt break. absoluteTolerance reached.")
					break
				elif self.which=="SM" and abs(bestL)<=absoluteTolerance:
					logger.log(18, "varopt break. absoluteTolerance reached.")
					break
				elif self.which=="LM" and abs(bestL)>=absoluteTolerance:
					logger.log(18, "varopt break. absoluteTolerance reached.")
					break
			if relativeTolerance is not None:
				#print("bestL",bestL)
				#print("previousBestL",previousBestL)
				if abs(bestL-previousBestL)<=abs(relativeTolerance*bestL):
					logger.log(18, "varopt break. relativeTolerance reached.")
					break
			if bestL==previousBestL:
				logger.log(18, "varopt break. no more opt.")
				break

		sweepTurnl = sweepTurni+1
		logger.log(18, f"varopt done. sweepTurnl == {sweepTurnl}")
		return bestL, sweepTurnl