import numpy as np

import floquet
import TLS

def singlePhotonGFunc(floquetSolver: floquet.floquetEigenSolver) -> np.ndarray:
	"""
	Compute single-photon Green's function for a periodically modulated TLS

	Args:
		floquetSlover: The solver of Floquet eigenvalues and eigenstates;
		inpFreq: The input frequencies of the input photon.

	Returns:
		gFunc_F: The Green's function in frequency domain
	"""
	L = np.array([[0.0, 0.0], [1.0, 0.0]])*np.sqrt(floquetSolver._tls._gamma)    # coupling operator
	LDagger = L.conj().T
	gBra = np.array([[0, 1]])
	gKet = gBra.conj().T
	L1to0 = []
	L0to1 = []
	floquetEigVals, floquetEigVecs = floquetSolver.floquetEigens
	for ii in range(floquetSolver._num_dt):
		excitedEigAtInsT = floquetEigVecs[ii][:][0].T
		L1to0Temp = gBra @ L @ excitedEigAtInsT
		L0to1Temp = excitedEigAtInsT.T @ LDagger @ gKet
		L1to0.append(L1to0Temp)
		L0to1.append(L0to1Temp)

	excitedEigVal = floquetEigVals[0]

	# time-domain Green's function
	gFunc_T = []

	for tt in range(floquetSolver._num_dt):
		gFuncFixedT = []
		for ss in range(floquetSolver._num_dt):
			if tt >= ss:
				gFuncEle = L1to0[tt]*np.exp(-1.0j*excitedEigVal*(tt - ss)*floquetSolver._dt)*L0to1[ss]
			else:
				gFuncEle = 0.0 + 0.0j
			gFuncFixedT.append(gFuncEle)
		gFunc_T.append(gFuncFixedT)


	# 2D Fourier transform to get frequency-domain Green's function
	gFunc_F1 = np.fft.fftshift(np.fft.fft(gFunc_T, axis=1), axes=1)/(floquetSolver._num_dt - 1)    # Fourier transform of input time s
	gFunc_F = np.fft.fftshift(np.fft.ifft(gFunc_F1, axis=0), axes=0)/(floquetSolver._num_dt - 1)    # inverse Fourier transform of output time t

	return gFunc_F

	# # FFT L0to1
	# L1to0new = np.array(L1to0)
	# L0to1fft = np.fft.fftshift(np.fft.fft(L0to1, axis=0), axes=0)/floquetSolver._num_dt
	# harmonics = 2*np.pi/floquetSolver._period*np.arange(-500, 501)
	# lorenzian = 1/(1.0j*(harmonics[np.newaxis, :] - inpFreq[:, np.newaxis]) + 1.0j*+ excitedEigVal)
	# L0to1fftConv = L0to1fft[np.newaxis, :, :]*lorenzian[:, :, np.newaxis]
	# L0to1Conv = np.fft.ifft(np.fft.ifftshift(L0to1fftConv, axes=1), axis=1)*floquetSolver._num_dt

	# gFunc_T = np.matmul(L1to0new[np.newaxis, :, :], L0to1Conv)[:, :, 0]
	# gFunc_F = np.fft.fftshift(np.fft.ifft(gFunc_T, axis=1), axes=1)

	# return gFunc_F



