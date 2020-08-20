import numpy as np
import scipy.linalg

import TLS
import drive

def singlePhotonSMatrix(tls: TLS.modulatedTLS, dts: np.array, cohDrive: drive.driveAb, tp: int) -> np.ndarray:
	"""
	Compute the single-photon S matrix element S(t',mu_out;t,mu_in).

	Args:
		tls: The two-level system under consideration;
		dts: The time-step array used to evaluate the propagator;
		cohDrive: The coherent drive applied to the TLS;
		tp: The width of the driving pulse.
	"""
	propagatorList = []    # The propagator list for each time step in 'dts'.
	inv_propagatorList = []
	for ii in range(dts.size - 1):
		dt = dts[ii + 1] - dts[ii]    # length of the time step
		tMean = (dts[ii] + dts[ii + 1])/2    # the mean time used to decide the value of drive
		Heff = np.array([[tls.hamiltonianEff(), cohDrive(tMean) + 0.0j], [cohDrive(tMean) + 0.0j, 0.0 + 0.0j]])
		propagatorList.append(scipy.linalg.expm(-1.0j*Heff*dt))
		inv_propagatorList.append(scipy.linalg.expm(1.0j*Heff*dt))


	cumulatedPropagatorList = []
	tempCumProp = np.eye(2, dtype=complex)
	cumulatedPropagatorList.append(tempCumProp)
	for ii in propagatorList:    # cumulated propagator list to calculate the time-dependent coupling operator
		tempCumProp = ii @ tempCumProp
		cumulatedPropagatorList.append(tempCumProp)

	inv_cumulatedPropagatorList = []
	inv_tempCumProp = np.eye(2, dtype=complex)
	inv_cumulatedPropagatorList.append(inv_tempCumProp)
	for ii in inv_propagatorList:    # cumulated propagator list to calculate the time-dependent coupling operator
		inv_tempCumProp = inv_tempCumProp @ ii    # note that this order is different, because (\sigma^\dagger\sigma) does not commute with (\sigma+\sigma^\dagger)
		inv_cumulatedPropagatorList.append(inv_tempCumProp)

	L = np.array([[0.0, 0.0], [1.0, 0.0]])*np.sqrt(tls._gamma/2)    # coupling operator
	propAtTp = cumulatedPropagatorList[tp] @ inv_cumulatedPropagatorList[int(dts.size/2)]   # U_{eff}(Tp,0)=U_{eff}(Tp,-infty)U_{eff}(-infty,0), this is doable since no drive for negative times
	sMatEle = []
	for tOut in range(dts.size):
		sMatEleTOut = []
		for tIn in range(dts.size):
			tildeL = cumulatedPropagatorList[int(dts.size/2)] @ inv_cumulatedPropagatorList[tOut] @ L @ cumulatedPropagatorList[tOut] @ inv_cumulatedPropagatorList[int(dts.size/2)]
			tildeLDagger = cumulatedPropagatorList[int(dts.size/2)] @ inv_cumulatedPropagatorList[tIn] @ L.conj().T @ cumulatedPropagatorList[tIn] @ inv_cumulatedPropagatorList[int(dts.size/2)]
			if tOut > tIn:
				sMatEleTOut.append((propAtTp @ tildeL @ tildeLDagger)[1][1])
			else:
				sMatEleTOut.append((propAtTp @ tildeLDagger @ tildeL)[1][1])
		sMatEle.append(sMatEleTOut)

	return sMatEle


def transmission(sinPhoSMat: np.ndarray, inpPhoFunc: drive.normInputPhotonProfile, dts: np.array) -> float:
	"""
	Compute the transmission of a input Gaussian single-photon pulse.

	Args:
		sinPhoSMat: The single-photon scattering matrix elements;
		inpPhoFunc: The input-photon profile function;
		dts: The time steps for both input and output photons.
	"""
	outProfile = []
	for tPrime in range(dts.size):
		gOutTPrime = 0.0 + 0.0j
		for t in range(dts.size):
			inputTime = dts[t]
			gOutTPrime += sinPhoSMat[tPrime][t]*inpPhoFunc(inputTime)*(dts[1] - dts[0])
		outProfile.append(gOutTPrime)

	return np.sum(np.abs(outProfile)**2)*(dts[1] - dts[0])


def transmissionVScentFreq(sinPhoSMat: np.array, spatialExtent: float, centralFreqArray: np.array, dts: np.array) -> np.array:
	"""
	Compute the transmission of a input Gaussian single-photon pulse as a function of central frequency of the photon.

	Args:
		sinPhoSMat: The single-photon scattering matrix elements;
		spatialExtent: The spatial extent of the input photon profile;
		centralFreqArray: The central frequency array for input photons;
		dts: The time steps for both input and output photons.
	"""
	trans = []

	for centralFreq in centralFreqArray:
		inputProf = drive.normInputPhotonProfile(spatialExtent, centralFreq)
		trans.append(transmission(sinPhoSMat, inputProf, dts))

	return trans







