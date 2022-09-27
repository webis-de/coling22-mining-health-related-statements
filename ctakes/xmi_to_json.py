import argparse
from collections import defaultdict
import json
import pathlib
from typing import Any, Dict, Iterator, List, Set, Tuple

from lxml import etree


def extract_umls_concepts(
    root: etree._ElementTree, nsmap: Dict[str, str]
) -> Dict[int, Any]:
    umls_concepts = {
        int(element.get(f"{{{nsmap['xmi']}}}id")): {
            "cui": element.get("cui"),
            "tui": element.get("tui"),
            "preferred": element.get("preferredText"),
        }
        for element in root.findall("refsem:UmlsConcept", nsmap)
    }
    entities = []
    for element in root.findall(".//textsem:*", nsmap):
        ontology_concepts = element.get("ontologyConceptArr")
        if ontology_concepts is None:
            continue
        entities.append(
            (
                int(element.get("begin")),
                int(element.get("end")),
                [int(umls_concept) for umls_concept in ontology_concepts.split(" ")],
            )
        )
    for entity in entities:
        begin, end, umls_ids = entity
        for umls_id in umls_ids:
            assert umls_concepts[umls_id].get("span") is None
            umls_concepts[umls_id]["span"] = (begin, end)
    concept_tuples = set()
    for concept_id, concept in list(umls_concepts.items()):
        concept_tuple = ((concept["cui"]), concept["tui"], concept["span"])
        if concept_tuple not in concept_tuples:
            concept_tuples.add(concept_tuple)
        else:
            del umls_concepts[concept_id]
    return umls_concepts


def match_words_and_concepts(
    word_spans: List[Tuple[int, int]], umls_concepts: Dict[int, Any]
) -> List[List[Tuple[str, str]]]:
    out: List[List[Tuple[str, str]]] = [[] for _ in range(len(word_spans))]
    umls_range_dict: Dict[range, Set[Tuple[str, str]]] = defaultdict(set)
    for concept in umls_concepts.values():
        umls_range_dict[range(*concept["span"])].add((concept["cui"], concept["tui"]))
    for idx, (begin, end) in enumerate(word_spans):
        for umls_range, umls_ids in umls_range_dict.items():
            if begin in umls_range or end - 1 in umls_range:
                out[idx].extend(list(umls_ids))
    return out


def parse_file(path: pathlib.Path) -> Dict[str, Any]:
    file_id = int(path.with_suffix("").with_suffix("").name)
    root = etree.parse(str(path)).getroot()
    nsmap = root.nsmap
    text_node = root.find("cas:Sofa", nsmap)
    assert text_node is not None
    text = text_node.get("sofaString")
    assert text
    word_spans = [
        (int(element.get("begin")), int(element.get("end")))
        for element in root.findall("syntax:WordToken", nsmap)
    ]
    words = [text[start:end] for start, end in word_spans]
    umls_concepts = extract_umls_concepts(root, nsmap)
    umls_id_list = match_words_and_concepts(word_spans, umls_concepts)
    umls_concept_values = list(umls_concepts.values())
    out = {
        "text": text,
        "words": words,
        "word_spans": word_spans,
        "umls_id_list": umls_id_list,
        "umls_concepts": umls_concept_values,
        "file_id": file_id,
    }
    return out


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--xmi_path", type=pathlib.Path, default="./output-dir")
    parser.add_argument("--output_path", type=pathlib.Path, default="./out.jsonl")

    args = parser.parse_args()

    paths: Iterator[pathlib.Path] = args.xmi_path.glob("*.xmi")

    out = []
    for path in paths:
        out.append(parse_file(path))
    out = sorted(out, key=lambda x: x["file_id"])
    with args.output_path.open("w", encoding="utf8") as file:
        for obj in out:
            file.write(json.dumps(obj) + "\n")


if __name__ == "__main__":
    main()
