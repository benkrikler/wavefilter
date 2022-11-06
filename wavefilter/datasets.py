from dataclasses import dataclass
from typing import Any, Callable, List, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from scipy.ndimage import convolve1d

from .pulse_functions import simple


@dataclass
class ToyData:
    length: int
    shape: float
    in_noise: float
    out_noise: float
    generator: Callable[[int], List[Tuple[int, float]]]
    max_truth_pulses: int = 10

    def __call__(self, n_samples: int) -> Tuple[npt.NDArray[float], Tuple[npt.NDArray[float], npt.NDArray[int]]]:
        pulse_length = int(10 * self.shape)
        padded_length = self.length + 2 * (pulse_length - 1)
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
        pulse /= pulse.max()
        y = convolve1d(y, pulse, origin=1 - pulse_length // 2)

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

    def __call__(self, n_samples: int) -> List[Tuple[int, float]]:
        t1 = np.random.normal(self.t1_mean, self.t1_std, size=n_samples).astype(int)
        t2 = t1 + np.random.randint(self.dt2_low, self.dt2_high, size=n_samples, dtype=int)
        a1, a2 = np.random.uniform(self.a_low, self.a_high, size=(2, n_samples))
        return [(t1, a1), (t2, a2)]


def to_pandas(
    data: npt.NDArray[float], truth_amps: npt.NDArray[float], truth_time: npt.NDArray[int], save: str = ""
) -> pd.DataFrame:
    dfs = [pd.DataFrame(i) for i in [data, truth_amps, truth_time]]
    df = pd.concat(dfs, keys=["data", "truth_amplitude", "truth_time"], axis=1)
    if save:
        df.to_pickle(save)
    return df


# TODO: use better type annotations than Any
def to_dataloader(data: npt.NDArray[float], **kwargs: Any) -> Any:
    # Add an extra dimension for the channel number
    data = data[:, np.newaxis, :]
    data = torch.tensor(data)
    ds = torch.utils.data.TensorDataset(data)
    return torch.utils.data.DataLoader(ds, **kwargs)


def generate_double_pulse_dataset(
    n_samples: int,
    t1_mean: float = 100,
    t1_std: float = 10,
    dt2_low: float = 60,
    dt2_high: float = 150,
    a_low: float = 10,
    a_high: float = 150,
    length: int = 1000,
    shape: float = 30,
    in_noise: float = 1.5,
    out_noise: float = 3,
) -> Tuple[npt.NDArray[float], Tuple[npt.NDArray[float], npt.NDArray[int]]]:

    generator = DoublePulses(
        t1_mean=t1_mean, t1_std=t1_std, dt2_low=dt2_low, dt2_high=dt2_high, a_low=a_low, a_high=a_high
    )
    builder = ToyData(length=length, shape=shape, in_noise=in_noise, out_noise=out_noise, generator=generator)
    data, truth = builder(n_samples)
    return data, truth


if __name__ == "__main__":
    import argparse
    import logging

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, required=True)
    parser.add_argument("--t1_mean", type=float, default=100)
    parser.add_argument("--t1_std", type=float, default=10)
    parser.add_argument("--dt2_low", type=float, default=100)
    parser.add_argument("--dt2_high", type=float, default=700)
    parser.add_argument("--a_low", type=float, default=10)
    parser.add_argument("--a_high", type=float, default=150)
    parser.add_argument("--length", type=int, default=1000)
    parser.add_argument("--shape", type=float, default=40)
    parser.add_argument("--in_noise", type=float, default=0.65)
    parser.add_argument("--out_noise", type=float, default=3)
    parser.add_argument(
        "--save", type=str, default="data_double_pulse-n_{n_samples}-in_{in_noise}-out_{out_noise}-shape_{shape}.pkl.gz"
    )
    args = vars(parser.parse_args())

    save = args.pop("save")
    save = save.format(**args)

    data, truth = generate_double_pulse_dataset(**args)
    to_pandas(data, *truth, save=save)
    logging.info(f"Wrote data to '{save}'")
