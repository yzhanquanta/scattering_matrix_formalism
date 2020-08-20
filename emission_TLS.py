import numpy as np
import scipy.linalg

import TLS
import drive

def propagatorEle_1g(tls: TLS.modulatedTLS, dts: np.array, cohDrive: drive.driveAb) -> np.array:
	"""
	Compute the integrand of P_{1,g}(infty), i.e., <t;g|U(Tp,0)|vac;g>.

	Args:
		tls: The two-level system under consideration;
		dts: The time-step array used to evaluate the propagator;
		cohDrive: The coherent drive applied to the TLS.
	"""
	propagatorList = []    # The propagator list for each time step in 'dts'.
	for ii in range(dts.size - 1):
		dt = dts[ii + 1] - dts[ii]    # length of the time step
		tMean = (dts[ii] + dts[ii + 1])/2    # the mean time used to decide the value of drive
		Heff = np.array([[tls.hamiltonianEff(), cohDrive(tMean) + 0.0j], [cohDrive(tMean) + 0.0j, 0.0 + 0.0j]])
		propagatorList.append(scipy.linalg.expm(-1.0j * Heff * dt))

	cumulatedPropagatorList = []
	tempCumProp = np.eye(2, dtype=complex)
	cumulatedPropagatorList.append(tempCumProp)
	for ii in propagatorList:    # cumulated propagator list to calculate the time-dependent coupling operator
		tempCumProp = np.dot(ii, tempCumProp)
		cumulatedPropagatorList.append(tempCumProp)

	L = np.array([[0.0, 0.0], [1.0, 0.0]])*np.sqrt(tls._gamma)    # coupling operator
	emissionMatEle = []
	for ii in cumulatedPropagatorList:
		tildeL = ii.conj().T @ L @ ii
		tempMatEle = (cumulatedPropagatorList[-1] @ tildeL)[1][1]
		# tempMatEle = (cumulatedPropagatorList[-1] @ L @ cumulatedPropagatorList[-1].conj().T)[1][1]
		emissionMatEle.append(tempMatEle)

	return emissionMatEle


def propagatorEle_0g(tls: TLS.modulatedTLS, dts: np.array, cohDrive: drive.driveAb) -> complex:
	"""
	Compute P_{0,g}(infty), i.e., <vac;g|U(Tp,0)|vac;g>.

	Args:
		tls: The two-level system under consideration;
		dts: The time-step array used to evaluate the propagator;
		cohDrive: The coherent drive applied to the TLS.
	"""
	propagatorList = []    # The propagator list for each time step in 'dts'.
	for ii in range(dts.size - 1):
		dt = dts[ii + 1] - dts[ii]    # length of the time step
		tMean = (dts[ii] + dts[ii + 1])/2    # the mean time used to decide the value of drive
		Heff = np.array([[tls.hamiltonianEff(), cohDrive(tMean) + 0.0j], [cohDrive(tMean) + 0.0j, 0.0 + 0.0j]])
		propagatorList.append(scipy.linalg.expm(-1.0j * Heff * dt))

	cumulatedProp = np.eye(2, dtype=complex)
	for ii in propagatorList:
		cumulatedProp = np.dot(ii, cumulatedProp)

	return cumulatedProp[1][1]


def insPropEle_0g(tls: TLS.modulatedTLS, dts: np.array, cohDrive: drive.driveAb, tp: float) -> np.array:
	"""
	Compute P_{0,g}(tau), i.e., <vac;g|U(tau,0)|vac;g>.

	Args:
		tls: The two-level system under consideration;
		dts: The time-step array used to evaluate the propagator;
		cohDrive: The coherent drive applied to the TLS;
		tp: The width of the driving pulse.
	"""
	propagatorList = []    # The propagator list for each time step in 'dts'.
	for ii in range(dts.size - 1):
		dt = dts[ii + 1] - dts[ii]    # length of the time step
		tMean = (dts[ii] + dts[ii + 1])/2    # the mean time used to decide the value of drive
		Heff = np.array([[tls.hamiltonianEff(), cohDrive(tMean) + 0.0j], [cohDrive(tMean) + 0.0j, 0.0 + 0.0j]])
		propagatorList.append(scipy.linalg.expm(-1.0j * Heff * dt))

	cumulatedPropagatorList = []
	tempCumProp = np.eye(2, dtype=complex)
	cumulatedPropagatorList.append(tempCumProp)
	for ii in propagatorList:    # cumulated propagator list to calculate the time-dependent coupling operator
		tempCumProp = np.dot(ii, tempCumProp)
		cumulatedPropagatorList.append(tempCumProp)

	insEmMatEle = []
	insEmMatEleTp = 0.0 + 0.0j
	for ii in range(dts.size):
		if dts[ii] < tp:
			insEmMatEle.append(cumulatedPropagatorList[ii][1][1])
			insEmMatEleTp = cumulatedPropagatorList[ii][1][1]
		else:
			insEmMatEle.append(insEmMatEleTp)

	return insEmMatEle


def insPropEle_0e(tls: TLS.modulatedTLS, dts: np.array, cohDrive: drive.driveAb, tp: float) -> np.array:
	"""
	Compute P_{0,e}(tau), i.e., <vac;e|U(tau,0)|vac;g>.

	Args:
		tls: The two-level system under consideration;
		dts: The time-step array used to evaluate the propagator;
		cohDrive: The coherent drive applied to the TLS;
		tp: The width of the driving pulse.
	"""
	propagatorList = []    # The propagator list for each time step in 'dts'.
	for ii in range(dts.size - 1):
		dt = dts[ii + 1] - dts[ii]    # length of the time step
		tMean = (dts[ii] + dts[ii + 1])/2    # the mean time used to decide the value of drive
		if dts[ii] < tp:
			Heff = np.array([[tls.hamiltonianEff(), cohDrive(tMean) + 0.0j], [cohDrive(tMean) + 0.0j, 0.0 + 0.0j]])
		else:
			Heff = np.array([[tls.hamiltonianEff(), 0.0 + 0.0j], [0.0 + 0.0j, 0.0 + 0.0j]])
		propagatorList.append(scipy.linalg.expm(-1.0j * Heff * dt))

	cumulatedPropagatorList = []
	tempCumProp = np.eye(2, dtype=complex)
	cumulatedPropagatorList.append(tempCumProp)
	for ii in propagatorList:    # cumulated propagator list to calculate the time-dependent coupling operator
		tempCumProp = np.dot(ii, tempCumProp)
		cumulatedPropagatorList.append(tempCumProp)

	insEmMatEle = []
	for ii in range(dts.size):
		insEmMatEle.append(cumulatedPropagatorList[ii][0][1])

	return insEmMatEle


def insPropEle_1g(tls: TLS.modulatedTLS, dts: np.array, cohDrive: drive.driveAb, tp: float) -> np.ndarray:
	"""
	Compute the integrand in P_{1,g}(tau), i.e., <vac;g|U(tau,0)|vac;g>.

	Args:
		tls: The two-level system under consideration;
		dts: The time-step array used to evaluate the propagator;
		cohDrive: The coherent drive applied to the TLS;
		tp: The width of the driving pulse;
	"""
	propagatorList = []    # The propagator list for each time step in 'dts'.
	inv_propagatorList = []
	for ii in range(dts.size - 1):
		dt = dts[ii + 1] - dts[ii]    # length of the time step
		tMean = (dts[ii] + dts[ii + 1])/2    # the mean time used to decide the value of drive
		if dts[ii] < tp:
			Heff = np.array([[tls.hamiltonianEff(), cohDrive(tMean) + 0.0j], [cohDrive(tMean) + 0.0j, 0.0 + 0.0j]])
		else:
			Heff = np.array([[tls.hamiltonianEff(), 0.0 + 0.0j], [0.0 + 0.0j, 0.0 + 0.0j]])
		propagatorList.append(scipy.linalg.expm(-1.0j*Heff*dt))
		inv_propagatorList.append(scipy.linalg.expm(1.0j*Heff*dt))

	cumulatedPropagatorList = []
	tempCumProp = np.eye(2, dtype=complex)
	cumulatedPropagatorList.append(tempCumProp)
	for ii in propagatorList:    # cumulated propagator list to calculate the time-dependent coupling operator
		tempCumProp = np.dot(ii, tempCumProp)
		cumulatedPropagatorList.append(tempCumProp)

	inv_cumulatedPropagatorList = []
	inv_tempCumProp = np.eye(2, dtype=complex)
	inv_cumulatedPropagatorList.append(inv_tempCumProp)
	for ii in inv_propagatorList:    # cumulated propagator list to calculate the time-dependent coupling operator
		inv_tempCumProp = np.dot(ii, inv_tempCumProp)
		inv_cumulatedPropagatorList.append(inv_tempCumProp)


	L = np.array([[0.0, 0.0], [1.0, 0.0]])*np.sqrt(tls._gamma)    # coupling operator
	insEmMatEle = []

	for tau in range(dts.size):
		insEmMatEleTau = []
		for ii in range(dts.size):
			tildeL = inv_cumulatedPropagatorList[tau - ii] @ L @ cumulatedPropagatorList[tau - ii]
			tempInsMatEle = np.dot(cumulatedPropagatorList[tau], tildeL)[1][1]
			if ii <= tau:
				insEmMatEleTau.append(tempInsMatEle)
			else:
				insEmMatEleTau.append(0.0 + 0.0j)
		insEmMatEle.append(insEmMatEleTau)

	return insEmMatEle


def insPropEle_1e(tls: TLS.modulatedTLS, dts: np.array, cohDrive: drive.driveAb, tp: float) -> np.ndarray:
	"""
	Compute the integrand in P_{1,g}(tau), i.e., <vac;g|U(tau,0)|vac;g>.

	Args:
		tls: The two-level system under consideration;
		dts: The time-step array used to evaluate the propagator;
		cohDrive: The coherent drive applied to the TLS;
		tp: The width of the driving pulse;
	"""
	propagatorList = []    # The propagator list for each time step in 'dts'.
	inv_propagatorList = []
	for ii in range(dts.size - 1):
		dt = dts[ii + 1] - dts[ii]    # length of the time step
		tMean = (dts[ii] + dts[ii + 1])/2    # the mean time used to decide the value of drive
		if dts[ii] < tp:
			Heff = np.array([[tls.hamiltonianEff(), cohDrive(tMean) + 0.0j], [cohDrive(tMean) + 0.0j, 0.0 + 0.0j]])
		else:
			Heff = np.array([[tls.hamiltonianEff(), 0.0 + 0.0j], [0.0 + 0.0j, 0.0 + 0.0j]])
		propagatorList.append(scipy.linalg.expm(-1.0j*Heff*dt))
		inv_propagatorList.append(scipy.linalg.expm(1.0j*Heff*dt))

	cumulatedPropagatorList = []
	tempCumProp = np.eye(2, dtype=complex)
	cumulatedPropagatorList.append(tempCumProp)
	for ii in propagatorList:    # cumulated propagator list to calculate the time-dependent coupling operator
		tempCumProp = np.dot(ii, tempCumProp)
		cumulatedPropagatorList.append(tempCumProp)

	inv_cumulatedPropagatorList = []
	inv_tempCumProp = np.eye(2, dtype=complex)
	inv_cumulatedPropagatorList.append(inv_tempCumProp)
	for ii in inv_propagatorList:    # cumulated propagator list to calculate the time-dependent coupling operator
		inv_tempCumProp = np.dot(ii, inv_tempCumProp)
		inv_cumulatedPropagatorList.append(inv_tempCumProp)


	L = np.array([[0.0, 0.0], [1.0, 0.0]])*np.sqrt(tls._gamma)    # coupling operator
	insEmMatEle = []

	for tau in range(dts.size):
		insEmMatEleTau = []
		for ii in range(dts.size):
			tildeL = inv_cumulatedPropagatorList[tau - ii] @ L @ cumulatedPropagatorList[tau - ii]
			tempInsMatEle = np.dot(cumulatedPropagatorList[tau], tildeL)[0][1]
			if ii <= tau:
				insEmMatEleTau.append(tempInsMatEle)
			else:
				insEmMatEleTau.append(0.0 + 0.0j)
		insEmMatEle.append(insEmMatEleTau)

	return insEmMatEle
