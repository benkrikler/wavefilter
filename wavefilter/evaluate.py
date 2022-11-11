import logging
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from tqdm.autonotebook import tqdm

from . import models


class CaptureActivations:
    def __init__(self) -> None:
        self.activations: Dict[str, torch.Tensor] = {}

    def __getitem__(self, name: str) -> torch.Tensor:
        return self.activations[name]

    def __call__(self, name: str) -> Callable[[Any, Any, Any], Any]:
        def hook(model: Any, input: Any, output: Any) -> None:
            self.activations[name] = output.detach()

        return hook

    def register(self, model: Any, *layers: str) -> None:
        for layer in layers:
            getattr(model, layer).register_forward_hook(self.__call__(layer))


# Based on https://medium.com/nerd-for-tech/convolution-neural-network-in-pytorch-81023e7de5b9
class TrainTester:
    def __init__(self, model: Any, optimizer: torch.optim.Optimizer, device: Any, loss: nn.modules.loss._Loss) -> None:
        self.model = model
        self.loss = loss
        # self.opt = torch.optim.Adam(self.model.parameters(), lr = self.lr)
        self.opt = optimizer
        self.train_loss: List[float] = []
        self.val_loss: List[float] = []
        self.train_acc: List[float] = []
        self.val_acc: List[float] = []
        self.device = device

    def batch_accuracy(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # output shape: [batch, 10]
        output = nn.functional.softmax(output, dim=1)
        output = output.argmax(1)
        acc = torch.sum(output == target) / output.shape[0]
        return acc.cpu() * 100

    def train_step(self, dataset: Any) -> None:
        self.model.train()
        batch_loss = []
        batch_acc = []
        for batch in dataset:
            inputs = batch[0].to(self.device)
            targets = inputs
            # targets = batch[1].to(self.device)
            self.opt.zero_grad()

            outputs = self.model(inputs)

            loss = self.loss(outputs, targets)
            loss.backward()
            self.opt.step()
            batch_loss.append(loss.item())
            batch_acc.append(self.batch_accuracy(outputs, targets))

        self.train_loss.append(np.mean(batch_loss))
        self.train_acc.append(np.mean(batch_acc))

    def validation_step(self, dataset: Any) -> None:
        self.model.train(False)
        batch_loss = []
        batch_acc = []
        with torch.no_grad():
            for batch in dataset:
                inputs = batch[0].to(self.device)
                targets = inputs
                # targets = batch[1].to(self.device)

                outputs = self.model(inputs)

                loss = self.loss(outputs, targets)
                batch_loss.append(loss.item())
                batch_acc.append(self.batch_accuracy(outputs, targets))

        self.val_loss.append(np.mean(batch_loss))
        self.val_acc.append(np.mean(batch_acc))

    def test_step(self, dataset: Any) -> None:
        self.model.train(False)
        batch_acc = []
        with torch.no_grad():
            for batch in dataset:
                inputs = batch[0].to(self.device)
                targets = inputs
                # targets = batch[1].to(self.device)

                outputs = self.model(inputs)
                batch_acc.append(self.batch_accuracy(outputs, targets))


ParamGroups = List[Dict[str, List[nn.parameter.Parameter]]]


def split_parameters(model: nn.Module, groups: List[str]) -> ParamGroups:
    """
    split_parameters(model, ["attend"])
    """
    parameters: ParamGroups = [dict(params=[]) for _ in range(len(groups) + 1)]
    for name, param in model.named_parameters():
        for i, group in enumerate(groups):
            if name.startswith(group):
                parameters[i]["params"].append(param)
                break
        else:
            parameters[-1]["params"].append(param)
    return parameters


def train_parallel_pulse_finder(
    data: Any,
    model: Any,
    param_groups: ParamGroups,
    lr_schedules: List[Callable[[int], float]],
    device: Any,
    conv_pf_name: str = "conv_pulse_finder",
    epochs: int = 300,
    learning_rate: float = 1e-3,
    start_incrementing: int = 40,
    epochs_per_increment: int = 12,
    change_per_increment: float = 0.05,
) -> Tuple[TrainTester, List[Any]]:

    optimiser = torch.optim.Adam(param_groups, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda=lr_schedules)
    # optimiser = torch.optim.Adam(model.parameters(), lr = learning_rate)
    train_test_runner = TrainTester(model, optimiser, device, nn.MSELoss())

    adapt_attention = isinstance(model.attend, models.ParallelWeightedModules)
    learning_rates = []

    for epoch in tqdm(range(epochs), desc="Epoch"):
        train_test_runner.train_step(data)
        if (
            adapt_attention
            and epoch > start_incrementing
            and (epoch + 1) % epochs_per_increment == 0
            and model.attend.get_weight(conv_pf_name) < 1
        ):
            model.attend.increment_weight(conv_pf_name, change_per_increment)
            logging.info(epoch, model.attend.module_weights)
        scheduler.step()
        learning_rates.append(scheduler.get_last_lr)
        # train_test_runner.validation_step(val_loader)
    # train_test_runner.test_step(test_loader)

    return train_test_runner, learning_rates
