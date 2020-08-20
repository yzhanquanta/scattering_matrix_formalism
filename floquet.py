import numpy as np
import scipy.linalg
from typing import Tuple

import TLS


class floquetEigenSolver:
	"""Define an object for soloving Floquet eigenvalues and eigenvectors."""
	def __init__(self, tls: TLS.modulatedTLS, period: float, num_dt: int, numPeriod: float):
		"""
		Create a Floquet eigen slover object.

		Args:
			tls: The periodic modulated two-level system under consideration;
			period: The period of the Hamiltonian;
			num_dt: The number of time steps in a period for numerical calculations.
		"""
		self._tls = tls
		self._period = period
		self._num_dt = num_dt
		self._dt = period/num_dt
		self._numPeriod = numPeriod


	@property
	def floquetEigens(self) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Compute the Floquet eigenvalues and eigenstates for a given periodic Hamiltonian.
		We use the fact that the Floquet state is the eigenstate of the "Floquet operator",
		with the quasi-energy being the exponent of the eigenvalue.

		Args:
			tls: The periodic modulated two-level system under consideration;
			period: The period of the Hamiltonian;
			num_dt: The number of time steps in a period for numerical calculations.

		Returns:
			floquet_eigenvals: The (possibly complex) Floquet eigenvalues;
			floquet_eigenvecs: The Floquet eigenstates.
		"""
		dt = self._numPeriod*self._period/self._num_dt

		# propagator with respect to the effective Hamiltonian for each time step in one period
		propList = []
		for ii in range(self._num_dt):
			Heff = np.array([[self._tls.hamiltonianTot((ii + 0.5)*dt), 0.0 + 0.0j], [0.0 + 0.0j, 0.0 + 0.0j]])    # effective Hamiltonian at each time step
			propTemp = scipy.linalg.expm(-1.0j*Heff*dt)
			propList.append(propTemp)

		# propagator for one period (Floquet operator)
		propAPeriod = np.eye(2, dtype=complex)
		for propEachTStep in propList:
			propAPeriod = propEachTStep @ propAPeriod

		# eigenvalues and eigenvectors of the Floquet operator, and the Floquet eigenvalues
		propEigVals, propEigVecs = np.linalg.eig(propAPeriod)    # eigenvalues and eigenvectors of the Floquet operator
		floquet_eigenvals_imag = -np.log(np.abs(propEigVals))/self._period    # the imaginary part, representing the decay, of the Floquet eigenvalue
		floquet_eigenvals_real = -np.arctan2(np.imag(propEigVals), np.real(propEigVals))/self._period    # the real part, representing the frequency, of the Floquet eigenvalue
		floquet_eigenvals = floquet_eigenvals_real - 1.0j*floquet_eigenvals_imag    # \omega_0 - i\gamma/2

		# Floquet eigenvectors for each time step in a period
		floquet_eigenvecs = [propEigVecs]    # at t=0 they are the same thing
		propEigVals_dt = (propEigVals[np.newaxis, :])**(1.0/self._num_dt)    # e^{-i\varepsion_0 dt}
		for propEachTStep in propList:
			floquet_eigenvecs_temp = (propEachTStep @ floquet_eigenvecs[-1])/propEigVals_dt    # U_{eff}(dt)\ket{\psi}=e^{-i\varepsilon_0 dt}\ket{\psi}
			floquet_eigenvecs.append(floquet_eigenvecs_temp)

		return floquet_eigenvals, floquet_eigenvecs




