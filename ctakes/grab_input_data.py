import argparse
import pathlib

import pandas as pd


def main(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--causenet_data_dir", type=pathlib.Path, required=True)

    args = parser.parse_args(args)

    causenet_data_dir: pathlib.Path = args.causenet_data_dir
    file_dir = pathlib.Path(__file__).resolve().parent
    input_dir = file_dir / "input"

    test_causenet = pd.read_pickle(causenet_data_dir / "test_causenet.pkl")
    sentence_test_causenet = pd.read_pickle(
        causenet_data_dir / "sentence_test_causenet.pkl"
    )

    relations = pd.unique(
        test_causenet.loc[:, ["cause", "effect"]].values.ravel()
    ).values
    sentence_relations = pd.unique(
        sentence_test_causenet.loc[:, "sentence"].values.ravel()
    ).values

    for idx, relation in enumerate(relations):
        if not relation.strip():
            continue
        with (input_dir / "phrase" / f"relation_{idx}.txt").open("w") as file:
            file.write(relation)

    for idx, relation in enumerate(sentence_relations):
        if not relation.strip():
            continue
        with (input_dir / "sentence" / f"relation_{idx}.txt").open("w") as file:
            file.write(relation)


if __name__ == "__main__":
    main()
