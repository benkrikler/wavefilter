import numpy as np
from dataclasses import dataclass
from .pulse_functions import simple
from typing import Callable, List, Tuple, Any
from scipy.ndimage import convolve1d


@dataclass
class ToyData:
    length: int
    shape: float
    in_noise: float
    out_noise: float
    generator: Callable[[], List[Tuple[int, float]]]
    max_truth_pulses: int = 10

    def __call__(self, n_samples: int) -> Tuple[Any, Tuple[Any, Any]]:
        pulse_length = 3 * self.shape
        padded_length = self.length + 2* (pulse_length - 1)
        t0 = pulse_length - 1

        data_dims = (n_samples, padded_length)
        y = np.zeros(data_dims, dtype=np.float32)

        truth_dims = (n_samples, self.max_truth_pulses)
        truth_amps = np.zeros(truth_dims, dtype=y.dtype)
        truth_time = np.zeros(truth_dims, dtype=int)

        for i_n in range(n_samples):
            pulses = self.generator(1)
            for i_p, (time, amplitude) in enumerate(pulses):
                if 0 > time or time >= self.length:
                    continue

                y[i_n, time + t0] = amplitude
                truth_amps[i_n, i_p] = amplitude
                truth_time[i_n, i_p] = time

        # Add input noise
        y += np.random.normal(scale=self.in_noise, size=data_dims)

        # Convolve with pulse shape
        pulse = simple(np.arange(pulse_length), self.shape)
        y = convolve1d(y, pulse, origin=pulse_length//2 - 1)

        # Add output noise
        y += np.random.normal(scale=self.out_noise, size=data_dims)

        # Trim bad regions from convolution
        y = y[:, pulse_length:-pulse_length]

        return y, (truth_amps, truth_time)


@dataclass
class DoublePulses:
    t1_mean: float
    t1_std: float
    dt2_low: float
    dt2_high: float
    a_low: float = 0
    a_high: float = 100

    def __call__(self, n_samples) -> List[Tuple[int,float]]:
        t1 = np.random.normal(self.t1_mean, self.t1_std, size=n_samples).astype(int)
        t2 = t1 + np.random.randint(self.dt2_low, self.dt2_high, size=n_samples, dtype=int)
        a1, a2 = np.random.uniform(self.a_low, self.a_high, size=(2, n_samples))
        return [(t1, a1), (t2, a2)]


def generate_double_pulse_dataset(n_samples, shape):
    generator = DoublePulses(t1_mean = 20, t1_std=10, dt2_low=60, dt2_high=150, a_low = 10, a_high=150)
    builder = ToyData(length=1000, shape=shape, in_noise=3, out_noise=3, generator=generator)
    data, truth = builder(n_samples)
    return data, truth
