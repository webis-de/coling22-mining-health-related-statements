import argparse
import pathlib

import pandas as pd


def main(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--causenet_data_dir", type=pathlib.Path, required=True)
    parser.add_argument("--datasets", type=str, nargs="*")
    parser.add_argument(
        "--save_dir",
        type=pathlib.Path,
        default=pathlib.Path(pathlib.Path(__file__).resolve().parent),
    )

    args = parser.parse_args(args)

    causenet_data_dir: pathlib.Path = args.causenet_data_dir
    save_dir: pathlib.Path = args.save_dir
    save_dir.mkdir(exist_ok=True)
    phrase_dir = save_dir / "phrase"
    phrase_dir.mkdir(exist_ok=True)
    sentence_dir = save_dir / "sentence"
    sentence_dir.mkdir(exist_ok=True)

    test_causenet = pd.read_pickle(causenet_data_dir / "test_causenet.pkl")
    sentence_test_causenet = pd.read_pickle(
        causenet_data_dir / "sentence_test_causenet.pkl"
    )
    if args.datasets:
        test_causenet = test_causenet.loc[test_causenet.dataset.isin(args.datasets)]
        sentence_test_causenet = sentence_test_causenet.loc[
            sentence_test_causenet.dataset.isin(args.datasets)
        ]

    relations = pd.unique(test_causenet.loc[:, ["cause", "effect"]].values.ravel())
    sentence_relations = pd.unique(
        sentence_test_causenet.loc[:, "sentence"].values.ravel()
    )

    for idx, relation in enumerate(relations):
        if not relation.strip():
            continue
        with (phrase_dir / f"relation_{idx}.txt").open("w") as file:
            file.write(relation)

    for idx, relation in enumerate(sentence_relations):
        if not relation.strip():
            continue
        with (sentence_dir / f"relation_{idx}.txt").open("w") as file:
            file.write(relation)


if __name__ == "__main__":
    main()
