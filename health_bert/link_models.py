import pathlib
import argparse
from typing import List


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_paths",
        type=pathlib.Path,
        nargs="+",
        required=True,
        help="paths to ckpt checkpoint model files to link to target directory",
    )

    parser.add_argument(
        "--target_dir",
        type=pathlib.Path,
        required=True,
        help="path to directory to link checkpoints to",
    )

    args = parser.parse_args()

    paths: List[pathlib.Path] = args.model_paths
    target_dir: pathlib.Path = args.target_dir

    for path in paths:
        model_name = path.parents[3].with_suffix(".ckpt").name
        (target_dir / model_name).symlink_to(path)


if __name__ == "__main__":
    main()
