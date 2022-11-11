import logging
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Mapping

import numpy as np
import torch
from torch import nn
from tqdm.autonotebook import tqdm

from . import models

logging.basicConfig(level=logging.INFO)


# Based on https://medium.com/nerd-for-tech/convolution-neural-network-in-pytorch-81023e7de5b9
class TrainTester:
    def __init__(
        self,
        model: Any,
        optimizer: torch.optim.Optimizer,
        device: Any,
        loss: nn.modules.loss._Loss,
        post_train_step_hooks: Iterable[Callable[[int, Mapping[str, List[float]]], None]] = tuple(),
    ) -> None:
        self.model = model
        self.loss = loss
        # self.opt = torch.optim.Adam(self.model.parameters(), lr = self.lr)
        self.opt = optimizer
        self.post_train_step_hooks = post_train_step_hooks
        self.train: Mapping[str, List[float]] = defaultdict(list)
        self.validate: Mapping[str, List[float]] = defaultdict(list)
        self.device = device

    def batch_accuracy(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # output shape: [batch, 10]
        output = nn.functional.softmax(output, dim=1)
        output = output.argmax(1)
        acc = torch.sum(output == target) / output.shape[0]
        return acc.cpu() * 100

    def train_step(self, epoch: int, dataset: Any) -> None:
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

        self.train["loss"].append(np.mean(batch_loss))
        self.train["acc"].append(np.mean(batch_acc))
        for hook in self.post_train_step_hooks:
            hook(epoch, self.train)

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

        self.validate["loss"].append(np.mean(batch_loss))
        self.validate["acc"].append(np.mean(batch_acc))

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
) -> TrainTester:

    optimiser = torch.optim.Adam(param_groups, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda=lr_schedules)

    def lr_hook(epoch: int, step_values: Mapping[str, List[float]]) -> None:
        scheduler.step()
        lrs = scheduler.get_last_lr()
        for i, lr in enumerate(lrs):
            step_values[f"lr_{i}"].append(lr)

    hooks = [lr_hook]

    has_adaptive_attention = isinstance(model.attend, models.ParallelWeightedModules)
    if has_adaptive_attention:

        def attend_hook(epoch: int, step_values: Mapping[str, List[float]]) -> None:
            epoch = epoch - start_incrementing
            if epoch > 0 and (epoch + 1) % epochs_per_increment == 0 and model.attend.get_weight(conv_pf_name) < 1:
                model.attend.increment_weight(conv_pf_name, change_per_increment)
                logging.info(epoch, model.attend.module_weights)
            step_values["pf_weight"].append(model.attend.get_weight(conv_pf_name))

        hooks.append(attend_hook)

    train_test_runner = TrainTester(model, optimiser, device, nn.MSELoss(), post_train_step_hooks=hooks)

    for epoch in tqdm(range(epochs), desc="Epoch"):
        train_test_runner.train_step(epoch, data)

    return train_test_runner
