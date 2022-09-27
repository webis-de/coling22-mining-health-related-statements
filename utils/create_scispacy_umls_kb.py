import argparse
import scispacy.linking_utils
import scispacy.candidate_generation
import scispacy.umls_utils
import pathlib
import json


def export_umls_to_jsonl(meta_path: str, output_path: str):
    "https://github.com/allenai/scispacy/blob/main/scripts/export_umls_json.py"

    concept_details = {}  # dictionary of concept_id -> {
    #                 'concept_id': str,
    #                 'canonical_name': str
    #                 'aliases': List[str]
    #                 'types': List[str]
    #                 'definition': str
    # }

    print("Reading concepts ... ")
    scispacy.umls_utils.read_umls_concepts(meta_path, concept_details)

    print("Reading types ... ")
    scispacy.umls_utils.read_umls_types(meta_path, concept_details)

    print("Reading definitions ... ")
    scispacy.umls_utils.read_umls_definitions(meta_path, concept_details)

    without_canonical_name_count = 0
    without_aliases_count = 0
    with_one_alias_count = 0
    with_more_than_one_alias_count = 0
    without_type_count = 0
    with_one_type_count = 0
    with_more_than_one_type_count = 0
    without_definition_count = 0
    with_definition_pref_source_count = 0
    with_definition_other_sources_count = 0
    for concept in concept_details.values():
        without_canonical_name_count += 1 if "canonical_name" not in concept else 0
        without_aliases_count += 1 if len(concept["aliases"]) == 0 else 0
        with_one_alias_count += 1 if len(concept["aliases"]) == 1 else 0
        with_more_than_one_alias_count += 1 if len(concept["aliases"]) > 1 else 0
        without_type_count += 1 if len(concept["types"]) == 0 else 0
        with_one_type_count += 1 if len(concept["types"]) == 1 else 0
        with_more_than_one_type_count += 1 if len(concept["types"]) >= 1 else 0
        without_definition_count += 1 if "definition" not in concept else 0
        with_definition_pref_source_count += (
            1 if concept.get("is_from_preferred_source") == "Y" else 0
        )
        with_definition_other_sources_count += (
            1 if concept.get("is_from_preferred_source") == "N" else 0
        )

    print(f"Number of concepts: {len(concept_details)}")
    print(
        f"Number of concepts without canonical name (one of the aliases will be used instead): "
        f"{without_canonical_name_count}"
    )
    print(f"Number of concepts with no aliases: {without_aliases_count}")
    print(f"Number of concepts with 1 alias: {with_one_alias_count}")
    print(f"Number of concepts with > 1 alias: {with_more_than_one_alias_count}")
    print(f"Number of concepts with no type: {without_type_count}")
    print(f"Number of concepts with 1 type: {with_one_type_count}")
    print(f"Number of concepts with > 1 type: {with_more_than_one_type_count}")
    print(f"Number of concepts with no definition: {without_definition_count}")
    print(
        f"Number of concepts with definition from preferred sources: {with_definition_pref_source_count}"
    )
    print(
        f"Number of concepts with definition from other sources: {with_definition_other_sources_count}"
    )

    print("Deleting unused fields and choosing a canonical name from aliases ... ")
    for concept in concept_details.values():

        # Some concepts have many duplicate aliases. Here we remove them.
        concept["aliases"] = list(set(concept["aliases"]))

        # if a concept doesn't have a canonical name, use the first alias instead
        if "canonical_name" not in concept:
            aliases = concept["aliases"]
            concept["canonical_name"] = aliases[0]
            del aliases[0]

        # deleting `is_from_preferred_source`
        if "is_from_preferred_source" in concept:
            del concept["is_from_preferred_source"]

    print("Exporting to the a jsonl file {} ...".format(output_path))
    with open(output_path, "w") as fout:

        for value in concept_details.values():
            fout.write(json.dumps(value) + "\n")
    print("DONE.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--meta_dir",
        type=pathlib.Path,
        required=True,
        help="path the META folder of umls subset",
    )
    parser.add_argument(
        "--kb_dir", type=pathlib.Path, required=True, help="path to save kb to"
    )

    args = parser.parse_args()

    assert args.meta_dir.exists()

    args.kb_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = args.kb_dir.joinpath("kb.jsonl")
    export_umls_to_jsonl(args.meta_dir, jsonl_path)

    kb = scispacy.linking_utils.KnowledgeBase(str(jsonl_path))
    scispacy.candidate_generation.create_tfidf_ann_index(str(args.kb_dir), kb)


if __name__ == "__main__":
    main()
