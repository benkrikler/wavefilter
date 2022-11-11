from typing import Any, Callable, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch


def plot_line(data: npt.NDArray[float], label: str, **kwargs: Any) -> None:
    scale = max(data.max(), -data.min())
    lines = plt.plot(data / scale, label=f"{label} ($\\times${scale:.01})", **kwargs)
    x = np.random.randint(len(data))
    y = data[x] / scale
    color = lines[0].get_color()
    plt.annotate(
        label,
        (x, y),
        xycoords="data",
        textcoords="offset points",
        xytext=(20, 20),
        arrowprops=dict(width=1, color=color),
        color=color,
        rotation=45,
    )


class InspectActivations:
    def __init__(self, device: Any) -> None:
        self.activations: Dict[str, npt.NDArray[float]] = {}
        self.device = device

    def __getitem__(self, name: str) -> npt.NDArray[float]:
        return self.activations[name]

    def add_hook(self, name: str) -> Callable[[Any, Any, Any], Any]:
        def hook(model: Any, input: Any, output: Any) -> None:
            self.activations[name] = output.detach().cpu().numpy()

        return hook

    def register(self, model: Any, *layers: str) -> None:
        for layer in layers:
            getattr(model, layer).register_forward_hook(self.add_hook(layer))

    def inspect(
        self,
        model: torch.nn.Module,
        data: Any,
        true_amp: npt.NDArray[float],
        true_time: npt.NDArray[int],
        indices: List[int],
    ) -> None:

        rows = len(indices)
        _, ax = plt.subplots(rows, 2, gridspec_kw={"width_ratios": [15, 10]}, figsize=(25, rows * 6.5))

        for i, idx in enumerate(indices):
            input = data[idx][0].to(self.device)
            self.activations["input"] = input.cpu().numpy()
            model(input)

            plt.sca(ax[i, 0])
            for label, values in self.activations.items():
                plot_line(values, label=label)
            plt.legend()

            plt.sca(ax[i, 1])
            self.plot_encoded(true_amp[i], true_time[i])

    def plot_encoded(self, true_amp: npt.NDArray[float], true_time: npt.NDArray[int]) -> None:
        max_amp = max(true_amp)
        encoded = self.activations["encoded"]
        scale = max_amp / encoded.max()
        plt.plot(encoded * scale, label=f" ($encoded \\times${scale:.01})")
        plt.vlines(true_time, 0, true_amp, color="green", label="truth")
        plt.scatter(true_time[true_amp != 0], true_amp[true_amp != 0], color="green")
        plt.legend()


# close_time = np.random.choice(np.where(truth[1][:, 1] < 250)[0], 5)
# far_time = np.random.choice(np.where(truth[1][:, 1] > 500)[0], 5)
# choices = np.concatenate((close_time, far_time))
