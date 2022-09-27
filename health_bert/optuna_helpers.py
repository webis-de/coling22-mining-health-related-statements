import argparse
import enum
from collections.abc import Iterable
from typing import Any, List, Optional, Sequence, Tuple, Union

import optuna
import pytorch_lightning as pl
from pytorch_lightning import callbacks


class OptunaArgumentError(Exception):
    pass


class OptunaKeyword(enum.Enum):
    CATEGORICAL = "categorical"
    INT = "int"
    UNIFORM = "uniform"
    LOGUNIFORM = "loguniform"
    DISCRETE_UNIFORM = "discrete_uniform"


class OptunaValue:
    def __init__(self, value: str):
        try:
            self.value = int(value)
        except ValueError:
            try:
                self.value = float(value)
            except ValueError:
                self.value = value

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self.value)


class OptunaArg:
    def __init__(self, args: List[Union[OptunaKeyword, OptunaValue]]):
        self.value = None
        self.keyword = None
        self.values = None
        error_message = (
            f"invalid optuna argument configuration, expected either one keyword "
            f"and multiple value or single value, got {list(type(arg) for arg in args)}"
        )
        if len(args) == 1:
            if not isinstance(args[0], OptunaValue):
                raise OptunaArgumentError(error_message)
            self.value = args[0].value
        else:
            keyword, *values = args
            if not isinstance(keyword, OptunaKeyword):
                raise OptunaArgumentError(error_message)
            if not all(isinstance(value, OptunaValue) for value in values):
                raise OptunaArgumentError(error_message)
            self.keyword = keyword
            self.values = [value.value for value in values]

    def sample(self, trial: optuna.Trial, key: str) -> Union[str, int, float]:
        if self.value is not None:
            return self.value
        assert self.keyword is not None and self.values is not None
        func = trial.__getattribute__(f"suggest_{self.keyword.value}")
        sample = func(key, *self.values)
        return sample

    @staticmethod
    def suggest_optuna_args(
        trial: optuna.Trial, args: argparse.Namespace
    ) -> argparse.Namespace:
        args_copy = argparse.Namespace(**vars(args))
        for key, argument in vars(args).items():
            if isinstance(argument, OptunaArg):
                args_copy.__setattr__(key, argument.sample(trial, key))
        return args_copy

    @staticmethod
    def parse(value: Any) -> Union[OptunaKeyword, OptunaValue]:
        try:
            return OptunaKeyword(value)
        except ValueError:
            return OptunaValue(value)


class OptunaArgumentParser(argparse.ArgumentParser):
    def parse_known_args(
        self,
        args: Optional[Sequence[str]] = None,
        namespace: Optional[argparse.Namespace] = None,
    ) -> Tuple[argparse.Namespace, List[str]]:
        namespace, args = super().parse_known_args(args, namespace)
        namespace = self.parse_optuna_args(namespace)
        return namespace, args

    @staticmethod
    def parse_optuna_args(namespace: argparse.Namespace) -> argparse.Namespace:
        for key, arg in vars(namespace).items():
            if isinstance(arg, (OptunaKeyword, OptunaValue)):
                arg = [arg]
            if isinstance(arg, Iterable) and all(
                isinstance(_arg, (OptunaKeyword, OptunaValue)) for _arg in arg
            ):
                namespace.__setattr__(key, OptunaArg(arg))
        return namespace


class OptunaPruningCallback(callbacks.EarlyStopping):
    def __init__(
        self,
        trial: optuna.Trial,
        monitor: Optional[str] = None,
        min_delta: float = 0.0,
        patience: int = 3,
        verbose: bool = False,
        mode: str = "min",
        strict: bool = True,
        check_finite: bool = True,
        stopping_threshold: Optional[float] = None,
        divergence_threshold: Optional[float] = None,
        check_on_train_epoch_end: Optional[bool] = None,
    ):
        super(OptunaPruningCallback, self).__init__(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            verbose=verbose,
            mode=mode,
            strict=strict,
            check_finite=check_finite,
            stopping_threshold=stopping_threshold,
            divergence_threshold=divergence_threshold,
            check_on_train_epoch_end=check_on_train_epoch_end,
        )
        self.trial = trial
        self.pruned = False

    def teardown(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        stage: Optional[str] = None,
    ) -> None:
        if self.pruned:
            raise optuna.exceptions.TrialPruned(
                f"Trial was pruned at step {trainer.global_step}"
            )

    def _run_early_stopping_check(self, trainer: "pl.Trainer") -> None:
        """Checks whether the early stopping condition is met and if so tells the trainer to stop the training."""
        logs = trainer.callback_metrics

        if (
            trainer.fast_dev_run
            or not self._validate_condition_metric(  # disable early_stopping with fast_dev_run
                logs
            )
        ):  # short circuit if metric not present
            return

        current = logs[self.monitor].squeeze()
        should_stop, reason = self._evaluate_stopping_criteria(current)
        step = trainer.global_step
        self.trial.report(current.item(), step)
        should_prune = self.trial.should_prune()
        self.pruned = should_prune
        if should_prune:
            reason = f"Trial was pruned at step {step}"

        # stop every ddp process if any world process decides to stop
        should_stop = trainer.training_type_plugin.reduce_boolean_decision(should_stop)
        trainer.should_stop = trainer.should_stop or should_stop
        if should_stop:
            self.stopped_epoch = trainer.current_epoch
        if reason and self.verbose:
            self._log_info(trainer, reason)
