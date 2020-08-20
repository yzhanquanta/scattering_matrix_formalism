import numpy as np
import abc

class driveAb(metaclass=abc.ABCMeta):
	@property
	@abc.abstractmethod
	def period(self) -> float:
		"""For getting the period of the drive."""
		raise NotImplementedError()

	def __call__(self, t: float) -> float:
		"""For calling the drive function."""
		raise NotImplementedError()


class sinDrive(driveAb):
	"""Define a sinusoidal drive class applied to the local system."""
	def __init__(self, mean: float, amp: float, phase: float, freq: float):
		"""
		Create a 'sinDrive' object.

		Args:
			mean: The mean value of the drive;
			amp: The amplitude of the drive;
			phase: The phase of the drive;
			freq: The angular frequency of the drive.
		"""
		self._mean = mean
		self._amp = amp
		self._phase = phase
		self._freq = freq

	@property
	def period(self) -> float:
		return 2*np.pi/self._freq

	def __call__(self, t: float) -> float:
		return self._mean + self._amp*np.sin(self._freq*t + self._phase)

class multiSinDrive(driveAb):
	"""Define a sinusoidal drive class applied to the local system."""
	def __init__(self, means: np.array, amps: np.array, phases: np.array, freqs: np.array):
		"""
		Create a 'multiSinDrive' object.

		Args:
			means: The mean values of all the driving harmonics;
			amps: The amplitudes of all the driving harmonics;
			phases: The phases of all the driving harmonics;
			freqs: The angular frequencies of all the driving harmonics.
		"""
		self._means = means
		self._amps = amps
		self._phases = phases
		self._freqs = freqs

	@property
	def period(self) -> float:
		return 2*np.pi/self._freqs[0]

	def __call__(self, t: float) -> float:
		modulation = 0.0
		for ii in range(self._means.size):
			modulation = modulation + self._means[ii] + self._amps[ii]*np.cos(self._freqs[ii]*t + self._phases[ii])
		return modulation


class pulseDrive(driveAb):
	"""Define a rectangular pulsed drive class applied to the local system"""
	def __init__(self, amp: float, width: float):
		"""
		Create a 'pulseDrive' object.

		Args:
			amp: The amplitude of the drive pulse;
			width: The width of the pulse.
		"""
		self._amp = amp
		self._width = width

	@property
	def period(self):
		return None

	def __call__(self, t: float) -> float:
		if (t > 0) and (t < self._width):
			return self._amp
		else:
			return 0.0


class normInputPhotonProfile(driveAb):
	"""Define the normalized input photon profile, which is a Gaussian function."""
	def __init__(self, spaExt: float, centFreq: float):
		"""
		Create a Gaussian function.

		Args:
			spaExt: The spatial extent of the input photon;
			centFreq: The central frequency of the input photon;
			t: The time instant.
		"""
		self._spaExt = spaExt
		self._centFreq = centFreq

	@property
	def period(self):
		return None

	def __call__(self, t: float) -> complex:
		return 1/(np.pi*self._spaExt**2)**(0.25)*np.exp(-t**2/(2*self._spaExt**2) + 1.0j*self._centFreq*t)


	