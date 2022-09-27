import argparse

import health_causenet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--causenet_path",
        type=str,
        default=health_causenet.constants.CAUSENET_JSON_PATH,
        help="bz2 zipped line json file of causenet extraction",
    )

    parser.add_argument(
        "--parquet_out_path",
        type=str,
        default=health_causenet.constants.CAUSENET_PARQUET_PATH,
        help="output path for parquet file",
    )

    parser.add_argument(
        "--split_size",
        type=int,
        default=0,
        help="number of relations to include per parquet file, "
        "if 0 all relations in 1 file",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    health_causenet.causenet.CauseNet.convert_to_parquet(
        args.causenet_path, args.parquet_out_path, args.split_size
    )


if __name__ == "__main__":
    main()
