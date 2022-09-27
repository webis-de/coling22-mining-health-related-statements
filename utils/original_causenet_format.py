import argparse
import bz2
import json
import pathlib
import re
from typing import Dict, List, Optional, TextIO

from tqdm import tqdm


class Writer:
    def __init__(self, base_dir: pathlib.Path, names: List[str]) -> None:
        base_dir.mkdir(parents=True, exist_ok=True)
        self.paths = {
            name: base_dir / f"{name}-health-causenet.jsonl.bz2" for name in names
        }
        self.files: Dict[str, Optional[TextIO]] = {name: None for name in names}

    def __enter__(self):
        for name, path in self.paths.items():
            self.files[name] = bz2.open(path, "wt")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for name, file in self.files.items():
            if file is not None:
                file.close()
                self.files[name] = None

    def write(self, name: str, text: str) -> None:
        file = self.files[name]
        if file is not None:
            file.write(text)


def main(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--causenet_jsonl_file", type=pathlib.Path, required=True)
    parser.add_argument("--health_causenet_tsv_file", type=pathlib.Path, required=True)
    parser.add_argument("--health_causenet_out_dir", type=pathlib.Path, required=True)

    args = parser.parse_args(args)

    health_causenet = {}
    with args.health_causenet_tsv_file.open("r") as health_causenet_file:
        health_causenet_file_iterator = iter(health_causenet_file)
        header = next(health_causenet_file_iterator).strip()
        names = header.split("\t")[2:]
        for line in health_causenet_file_iterator:
            split = line.strip().split("\t")
            cause, effect = split[:2]
            cause = cause.strip()
            effect = effect.strip()
            health_causenet[(cause, effect)] = [value == "True" for value in split[2:]]

    num_statements = {name: 0 for name in names}

    with bz2.open(args.causenet_jsonl_file, mode="rt") as json_line_file:
        writer = Writer(args.health_causenet_out_dir, names)
        with writer:
            for jsonl_line in tqdm(json_line_file, total=11609890):
                relation_dict = json.loads(jsonl_line)
                relation = relation_dict["causal_relation"]
                cause = relation["cause"]["concept"].replace("_", " ").strip()
                cause = re.sub(r" +", r" ", cause)
                effect = relation["effect"]["concept"].replace("_", " ").strip()
                effect = re.sub(r" +", r" ", effect)
                if "support" not in relation_dict:
                    pattern_set = set()
                    for source in relation_dict["sources"]:
                        source_type = source["type"]
                        if source_type in ("wikipedia_list", "wikipedia_infobox"):
                            continue
                        if pattern_set is not None:
                            pattern_set.add(source["payload"]["path_pattern"])
                    relation_dict["support"] = len(pattern_set)
                if (cause, effect) in health_causenet:
                    for name, health_related in zip(
                        names, health_causenet[(cause, effect)]
                    ):
                        if health_related:
                            num_statements[name] += 1
                            writer.write(name, json.dumps(relation_dict) + "\n")

    print(json.dumps(num_statements, indent=2))


if __name__ == "__main__":
    main(
        [
            "--causenet_jsonl_file",
            "/mnt/ceph/storage/data-in-progress/data-research/web-search/"
            "health-question-answering/causenet-data/causality-graphs/integration/"
            "causenet-full.jsonl.bz2",
            "--health_causenet_tsv_file",
            "/mnt/ceph/storage/data-in-progress/data-research/web-search/"
            "health-question-answering/causenet/health-causenet.tsv",
            "--health_causenet_out_dir",
            "/mnt/ceph/storage/data-in-progress/data-research/web-search/"
            "health-question-answering/health-causenet/",
        ]
    )

