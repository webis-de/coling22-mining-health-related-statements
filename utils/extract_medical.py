import argparse
import pathlib

import pandas as pd
import tqdm

import health_causenet
import health_causenet.causenet


def parse_args(args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "func",
        type=str,
        choices=list(
            health_causenet.causenet.CauseNet.MEDICAL_CONCEPT_EXTRACTION_FUNCS.keys()
        ),
        help="name of function to determine if relation is medical",
    )

    parser.add_argument(
        "--corpora",
        type=str,
        nargs="+",
        default=[],
        choices=(
            "pubmed",
            "pubmed_central",
            "textbook",
            "encyclopedia",
            "umls",
            "encyclopedia_umls",
        ),
        help="corpora to use for termhood scores",
    )

    parser.add_argument(
        "--n_gram_size",
        type=int,
        nargs=2,
        default=(1, 1),
        help="number of n grams to use for termhood scores",
    )

    parser.add_argument(
        "--p",
        type=str,
        default=1,
        help="value for generalized mean for multi-word termhood scores",
    )

    parser.add_argument(
        "--jaccard_threshold",
        type=float,
        default=1,
        help="jaccard similarity treshold for quickumls",
    )

    parser.add_argument(
        "--umls_subset",
        type=str,
        choices=("mesh", "mesh_syn", "mesh_meta_syn", "umls"),
        default="mesh",
        help="subset of UMLS to use for quickumls",
    )

    parser.add_argument(
        "--st21pv",
        type=int,
        choices=(0, 1),
        help="use st21pv subset of UMLS semantic types",
    )

    parser.add_argument(
        "--causenet_path",
        type=str,
        default=health_causenet.constants.CAUSENET_PARQUET_PATH,
        help="path to parquet dataframe file",
    )

    parser.add_argument(
        "-P",
        "--num_processes",
        type=int,
        default=1,
        help="number of parallel processes",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite already extracted parquet files",
    )

    args = parser.parse_args(args)

    return args


def main(args=None):
    args = parse_args(args)

    causenet_paths = pathlib.Path(args.causenet_path)
    if causenet_paths.is_dir():
        causenet_paths = causenet_paths.glob("causenet_*.parquet")
    else:
        causenet_paths = [causenet_paths]

    kwargs = {}
    if args.func == "quickumls":
        kwargs["umls_subset"] = args.umls_subset
        kwargs["jaccard_threshold"] = args.jaccard_threshold
        kwargs["st21pv"] = bool(args.st21pv)
        func_name = (
            f"{args.func}-{args.umls_subset}-{args.jaccard_threshold}-{args.st21pv}"
        )
    else:
        cf = health_causenet.causenet.load_cf(
            pubmed="pubmed" in args.corpora,
            pubmed_central="pubmed_central" in args.corpora,
            textbook="textbook" in args.corpora,
            encyclopedia="encyclopedia" in args.corpora,
            umls="umls" in args.corpora,
            encyclopedia_umls="encyclopedia_umls" in args.corpora,
        )
        kwargs["cf"] = cf
        kwargs["p"] = float(args.p)
        kwargs["n_gram_size"] = args.n_gram_size
        func_name = "-".join(
            [
                args.func,
                "_".join(args.corpora),
                f"({args.n_gram_size[0]}_{args.n_gram_size[1]})",
                str(args.p),
            ]
        )

    for causenet_path in causenet_paths:
        if (
            causenet_path.with_name(
                causenet_path.name.replace("causenet", func_name)
            ).exists()
            and not args.overwrite
        ):
            continue

        causenet = pd.read_parquet(causenet_path)

        causenet = health_causenet.causenet.CauseNet.is_medical(
            causenet, args.func, args.num_processes, **kwargs
        )

        causenet.to_parquet(
            str(
                causenet_path.with_name(
                    causenet_path.name.replace("causenet", func_name)
                )
            )
        )


if __name__ == "__main__":
    main()
