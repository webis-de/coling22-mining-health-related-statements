import argparse

import optuna
import pytorch_lightning as pl
from pytorch_lightning import callbacks as pl_callbacks

from health_bert import data, health_bert, optuna_helpers, util


def objective(trial: optuna.Trial, args: argparse.Namespace) -> float:

    args = optuna_helpers.OptunaArg.suggest_optuna_args(trial, args)

    pl.seed_everything(args.seed)

    checkpoint_kwargs = util.parse_arguments(
        pl_callbacks.ModelCheckpoint, args, ignore_args=["train_time_interval"]
    )
    checkpoint = pl_callbacks.ModelCheckpoint(**checkpoint_kwargs)

    pruning_kwargs = util.parse_arguments(
        optuna_helpers.OptunaPruningCallback, args, ignore_args=["trial"]
    )
    early_stop = optuna_helpers.OptunaPruningCallback(trial, **pruning_kwargs)

    data_kwargs = util.parse_arguments(data.HealthBertDatamodule, args)
    datamodule = data.HealthBertDatamodule(**data_kwargs)

    callbacks = [early_stop, checkpoint]
    if args.monitor_gpu_stats:
        device_stats = pl_callbacks.DeviceStatsMonitor()
        callbacks.append(device_stats)

    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks)
    model = health_bert.HealthBert(args)

    fit_kwargs = dict(datamodule=datamodule)
    if args.resume_checkpoint is not None:
        fit_kwargs["ckpt_path"] = args.resume_checkpoint

    trainer.fit(model, **fit_kwargs)

    return early_stop.best_score


def create_argument_parser() -> argparse.ArgumentParser:
    parser = optuna_helpers.OptunaArgumentParser()

    conflict_tracker = util.ArgumentConflictSolver()
    argparse._ActionsContainer.add_argument = conflict_tracker.catch_conflicting_args(
        argparse._ActionsContainer.add_argument
    )

    parser = util.add_argparse_args(pl.Trainer, parser)

    group = parser.add_argument_group("Monitor")
    group.add_argument(
        "--monitor_gpu_stats",
        action="store_true",
        dest="monitor_gpu_stats",
        help="toggle to monitor gpu status in logger",
    )

    parser = util.add_argparse_args(pl_callbacks.ModelCheckpoint, parser)
    parser = util.add_argparse_args(optuna_helpers.OptunaPruningCallback, parser)
    parser = util.add_argparse_args(data.HealthBertDatamodule, parser)
    parser = health_bert.HealthBert.add_model_specific_args(parser)

    group = parser.add_argument_group("Optuna")
    group.add_argument(
        "--n_trials",
        type=int,
        default=1,
        help=(
            "The number of trials. If this argument is set to None, there is no"
            " limitation on the number of trials. If timeout is also set to None, the"
            " study continues to create trials until it receives a termination signal"
            " such as Ctrl+C or SIGTERM, default 1"
        ),
    )
    group.add_argument(
        "--pruning",
        dest="pruning",
        action="store_true",
        help="if toggled activates experiment pruning",
    )
    group.add_argument(
        "--timeout",
        type=int,
        default=None,
        help=(
            "Stop study after the given number of second(s). If this argument is set"
            "to None, the study is executed without time limitation. If n_trials is"
            " also set to None, the study continues to create trials until it receives"
            " a termination signal such as Ctrl+C or SIGTERM, default None"
        ),
    )

    group = parser.add_argument_group("Global")
    group.add_argument(
        "--seed", type=int, default=None, help="seed for random number generators"
    )

    group = parser.add_argument_group("Resume")
    group.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="path to checkpoint to resume for training",
    )

    conflict_tracker.resolve_conflicting_args(
        group,
        {
            "--monitor": "quantity to monitor for early stopping, checkpointing "
            "and lr scheduling, default None",
            "--verbose": "verbosity mode, default False",
            "--mode": (
                "one of {min, max}, dictates if early stopping and checkpointing "
                "considers maximum or minimum of monitored quantity"
            ),
        },
    )
    for option_string, actions in conflict_tracker.conflicting_args.items():
        if option_string not in parser._option_string_actions:
            raise argparse.ArgumentError(
                actions.pop(), "missing global argument for conflicting argument"
            )

    return parser


def main():

    parser = create_argument_parser()
    args = parser.parse_args()

    pruner = (
        optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()
    )

    study = optuna.create_study(
        direction="maximize" if args.mode == "max" else "minimize", pruner=pruner
    )
    study.optimize(
        lambda trial: objective(trial, args),
        n_trials=args.n_trials,
        timeout=args.timeout,
    )

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == "__main__":
    main()
