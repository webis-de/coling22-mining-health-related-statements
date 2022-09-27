import argparse
from collections import defaultdict
import json
import pathlib
from typing import Any, Dict, Iterator, List, Set, Tuple

import nltk

semtype_to_cui = {
    "aapp": "T116",
    "acab": "T020",
    "acty": "T052",
    "aggp": "T100",
    "amas": "T087",
    "amph": "T011",
    "anab": "T190",
    "anim": "T008",
    "anst": "T017",
    "antb": "T195",
    "arch": "T194",
    "bacs": "T123",
    "bact": "T007",
    "bdsu": "T031",
    "bdsy": "T022",
    "bhvr": "T053",
    "biof": "T038",
    "bird": "T012",
    "blor": "T029",
    "bmod": "T091",
    "bodm": "T122",
    "bpoc": "T023",
    "bsoj": "T030",
    "celc": "T026",
    "celf": "T043",
    "cell": "T025",
    "cgab": "T019",
    "chem": "T103",
    "chvf": "T120",
    "chvs": "T104",
    "clas": "T185",
    "clna": "T201",
    "clnd": "T200",
    "cnce": "T077",
    "comd": "T049",
    "crbs": "T088",
    "diap": "T060",
    "dora": "T056",
    "drdd": "T203",
    "dsyn": "T047",
    "edac": "T065",
    "eehu": "T069",
    "elii": "T196",
    "emod": "T050",
    "emst": "T018",
    "enty": "T071",
    "enzy": "T126",
    "euka": "T204",
    "evnt": "T051",
    "famg": "T099",
    "ffas": "T021",
    "fish": "T013",
    "fndg": "T033",
    "fngs": "T004",
    "food": "T168",
    "ftcn": "T169",
    "genf": "T045",
    "geoa": "T083",
    "gngm": "T028",
    "gora": "T064",
    "grpa": "T102",
    "grup": "T096",
    "hcpp": "T068",
    "hcro": "T093",
    "hlca": "T058",
    "hops": "T131",
    "horm": "T125",
    "humn": "T016",
    "idcn": "T078",
    "imft": "T129",
    "inbe": "T055",
    "inch": "T197",
    "inpo": "T037",
    "inpr": "T170",
    "irda": "T130",
    "lang": "T171",
    "lbpr": "T059",
    "lbtr": "T034",
    "mamm": "T015",
    "mbrt": "T063",
    "mcha": "T066",
    "medd": "T074",
    "menp": "T041",
    "mnob": "T073",
    "mobd": "T048",
    "moft": "T044",
    "mosq": "T085",
    "neop": "T191",
    "nnon": "T114",
    "npop": "T070",
    "nusq": "T086",
    "ocac": "T057",
    "ocdi": "T090",
    "orch": "T109",
    "orga": "T032",
    "orgf": "T040",
    "orgm": "T001",
    "orgt": "T092",
    "ortf": "T042",
    "patf": "T046",
    "phob": "T072",
    "phpr": "T067",
    "phsf": "T039",
    "phsu": "T121",
    "plnt": "T002",
    "podg": "T101",
    "popg": "T098",
    "prog": "T097",
    "pros": "T094",
    "qlco": "T080",
    "qnco": "T081",
    "rcpt": "T192",
    "rept": "T014",
    "resa": "T062",
    "resd": "T075",
    "rnlw": "T089",
    "sbst": "T167",
    "shro": "T095",
    "socb": "T054",
    "sosy": "T184",
    "spco": "T082",
    "tisu": "T024",
    "tmco": "T079",
    "topp": "T061",
    "virs": "T005",
    "vita": "T127",
    "vtbt": "T010",
}


def match_words_and_concepts(
    word_spans: List[Tuple[int, int]], umls_concepts: List[Dict[str, Any]]
) -> List[List[Tuple[str, str]]]:
    out: List[List[Tuple[str, str]]] = [[] for _ in range(len(word_spans))]
    umls_range_dict: Dict[range, Set[Tuple[str, str]]] = defaultdict(set)
    for concept in umls_concepts:
        umls_range_dict[range(*concept["span"])].add((concept["cui"], concept["tui"]))
    for idx, (begin, end) in enumerate(word_spans):
        for umls_range, umls_ids in umls_range_dict.items():
            if begin in umls_range or end - 1 in umls_range:
                out[idx].extend(list(umls_ids))
    return out


def parse_positional_information(pos_infos: str) -> List[Tuple[int, int]]:
    spans = []
    pos_infos = pos_infos.replace("[", "").replace("]", "").replace(";", ",")
    for pos_info in pos_infos.split(","):
        start, length = pos_info.split("/")
        start = int(start)
        length = int(length)
        spans.append((start, start + length))
    return spans


def parse_lines(text: str, mmi_lines: List[str]) -> Dict[str, Any]:
    tokenizer = nltk.TreebankWordTokenizer()
    umls_concepts = []
    word_spans = list(tokenizer.span_tokenize(text))
    words = [text[start:end] for start, end in word_spans]
    for mmi_line in mmi_lines:
        split = mmi_line.split("|")
        *_, preferred, cui, sem_types, _, _, pos_infos, _ = split
        spans = parse_positional_information(pos_infos)
        sem_types = sem_types.replace("[", "").replace("]", "").split(",")
        tuis = [semtype_to_cui[sem_type] for sem_type in sem_types]
        for span in spans:
            for tui in tuis:
                umls_concepts.append(
                    {"cui": cui, "tui": tui, "preferred": preferred, "span": span}
                )
    umls_id_list = match_words_and_concepts(word_spans, umls_concepts)
    out = {
        "text": text,
        "words": words,
        "word_spans": word_spans,
        "umls_id_list": umls_id_list,
        "umls_concepts": umls_concepts,
    }
    return out


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mmi_path", type=pathlib.Path, default="./output-dir/relations.txt"
    )
    parser.add_argument(
        "--text_path", type=pathlib.Path, default="./input-dir/relations.txt"
    )
    parser.add_argument("--output_path", type=pathlib.Path, default="./out.jsonl")

    args = parser.parse_args()

    out = []
    mmi_lines = []
    num_lines = 0
    mmi_idx = -float("inf")
    mmi_line = ""
    with args.text_path.open("r") as text_file:
        with args.mmi_path.open("r") as mmi_file:
            iterator = iter(mmi_file)
            for text_line in text_file:
                num_lines += 1
                text_line = text_line.strip()
                text_idx, text_line = text_line.split("|")
                text_idx = int(text_idx)
                while True:
                    if mmi_idx > text_idx:
                        break
                    elif mmi_idx == text_idx:
                        mmi_lines.append(mmi_line)
                    try:
                        mmi_line = next(iterator)
                    except StopIteration:
                        break
                    mmi_line = mmi_line.strip()
                    split = mmi_line.split("|")
                    if len(split) < 2:
                        mmi_idx = -float("inf")
                        continue
                    mmi_idx = int(split[0])
                out.append(parse_lines(text_line, mmi_lines))
                mmi_lines = []
    assert len(out) == num_lines
    with args.output_path.open("w", encoding="utf8") as file:
        for obj in out:
            file.write(json.dumps(obj) + "\n")


if __name__ == "__main__":
    main()
