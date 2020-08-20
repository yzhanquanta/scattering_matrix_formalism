import numpy as np
import drive

class modulatedTLS:
	"""Define a modulated two-level system class."""
	def __init__(self, omega_0: float, gamma: float, Delta: drive.driveAb):
		"""
		Create a 'modulatedTLS' object.

		Args:
			omega_0: Unmodulated resoncance frequency of the TLS;
			gamma: The decay rate of the TLS;
			Delta: The drive applied to the TLS, which could be a periodic modulated or a pulse.
		"""
		self._omega_0 = omega_0
		self._gamma = gamma
		self._Delta = Delta

	@property
	def period(self) -> float:
		return self._Delta.period    # The TLS has the same period as the drive.

	def hamiltonianEff(self) -> complex:
		"""Time-independent part of effective Hamiltonian of the system."""
		return (self._omega_0 - 0.5j*self._gamma)

	def hamiltonianTot(self, t: float) -> complex:
		"""Time-dependent effective Hamiltonian of the system."""
		return (self._omega_0 + self._Delta(t) - 0.5j*self._gamma)