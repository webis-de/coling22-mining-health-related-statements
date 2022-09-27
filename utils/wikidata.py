import json
import os
import re
import time
from typing import Any, Dict, Optional
import urllib.parse

import pandas as pd
import requests
import tqdm

from health_causenet import constants

# SELECT ?cause ?causeLabel ?effect ?effectLabel WHERE {
#   {
#     SELECT DISTINCT ?cause ?effect WHERE {
#       VALUES ?medical {
#         wd:Q39833         # microorganism
#         wd:Q174211        # organic compound
#         wd:Q796194        # medical procedure
#         wd:Q1132455       # hazard
#         wd:Q2826767       # disease causative agent
#         wd:Q5850078       # etiology
#         wd:Q7189713       # physiological condition
#         wd:Q86746756      # medicinal product
#         wd:Q87075524      # health risk
#       }
#       ?cause (wdt:P1542|^wdt:P828) ?effect.
#       ?effect ((wdt:P460|wdt:P31|wdt:P1269|wdt:P279|wdt:P171|wdt:P361|wdt:P5642)/((wdt:P279|wdt:P171|wdt:P361|wdt:P5642)*)) ?medical.
#       hint:Prior hint:gearing "forward".
#       ?cause ((wdt:P460|wdt:P31|wdt:P1269|wdt:P279|wdt:P171|wdt:P361|wdt:P5642)/((wdt:P279|wdt:P171|wdt:P361|wdt:P5642)*)) ?medical.
#       hint:Prior hint:gearing "forward".
#     }
#   }
#   SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
# }


def parse_json(data):
    out = []
    for row in data:
        out.append(
            [
                row["cause"]["value"],
                row["causeLabel"]["value"],
                row["effect"]["value"],
                row["effectLabel"]["value"],
            ]
        )
    return pd.DataFrame(out, columns=["cause", "causeLabel", "effect", "effectLabel"])


def main():

    sleep_time = 15
    num_attempts = 5
    overwrite = False
    cause_effect_path = os.path.join(constants.WIKIDATA_PATH, "cause-effect.csv")
    cause_medical_effect_medical_path = os.path.join(
        constants.WIKIDATA_PATH, "cause-medical-effect-medical.json"
    )

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36"
    }

    url = "https://query.wikidata.org/sparql?query={query}&format=json"
    medical_entities = [
        # "wd:Q5",  # human !! query timeout
        "wd:Q764",  # fungus
        "wd:Q8054",  # protein
        "wd:Q39833",  # microorganism
        # "wd:Q178694",  # heredity
        # "wd:Q181394",  # nutrient
        "wd:Q289472",  # biogenic substance
        "wd:Q796194",  # medical procedure
        "wd:Q2826767",  # disease causative agent
        # "wd:Q2996394",  # biological process
        "wd:Q5850078",  # etiology
        "wd:Q7189713",  # physiological condition
        # "wd:Q15788410",  # state of consciousness
        "wd:Q86746756",  # medicinal product
        # "wd:Q87075524",  # health risk
    ]
    relations = ["?cause", "?effect"]
    tail_predicates = [
        # "wdt:P460",  # said to be the same as
        "wdt:P31",  # instance of
        # "wdt:P1269",  # facet of
    ]
    chain_predicates = [
        "wdt:P279",  # subclass of
        "wdt:P171",  # parent taxon
        "wdt:P5642",  # risk factor
    ]
    if os.path.exists(cause_effect_path) and not overwrite:
        cause_effect = pd.read_csv(cause_effect_path)
    else:
        all_query = (
            "SELECT ?cause ?causeLabel ?effect ?effectLabel WHERE {"
            "{"
            "  SELECT DISTINCT ?cause ?effect WHERE {"
            "    ?cause (wdt:P1542|^wdt:P828) ?effect."
            "  }"
            "}"
            "SERVICE wikibase:label { bd:serviceParam "
            'wikibase:language "[AUTO_LANGUAGE],en". }'
            "}"
        )
        response = requests.get(
            url.format(query=urllib.parse.quote_plus(all_query)), headers=headers
        )
        all_cause_effect = json.loads(response.content.decode())["results"]["bindings"]
        cause_effect = parse_json(all_cause_effect)
        cause_effect.to_csv(cause_effect_path, index=False)

    query = (
        "SELECT ?cause ?causeLabel ?effect ?effectLabel WHERE {{"
        "{{"
        "  SELECT DISTINCT ?cause ?effect WHERE {{"
        "    ?cause (wdt:P1542|^wdt:P828) ?effect."
        "    {relation} (({tail_predicates})?/"
        "    (({chain_predicates})*)) {medical_entity}."
        '    hint:Prior hint:gearing "forward".'
        "  }}"
        "}}"
        "SERVICE wikibase:label {{ bd:serviceParam "
        'wikibase:language "[AUTO_LANGUAGE],en". }}'
        "}}"
    )

    cause_medical_effect_medical: Optional[Dict[str, Any]] = None
    if os.path.exists(cause_medical_effect_medical_path):
        with open(cause_medical_effect_medical_path, "r") as file:
            cause_medical_effect_medical = json.load(file)
    if (
        cause_medical_effect_medical is None
        or cause_medical_effect_medical["tail_predicates"] != tail_predicates
        or cause_medical_effect_medical["chain_predicates"] != chain_predicates
    ):
        cause_medical_effect_medical = {relation: {} for relation in relations}
        cause_medical_effect_medical["tail_predicates"] = tail_predicates
        cause_medical_effect_medical["chain_predicates"] = chain_predicates
    pg = tqdm.tqdm(total=len(relations) * len(medical_entities))

    for relation in relations:
        for medical_entity in medical_entities:
            if (
                not overwrite
                and medical_entity in cause_medical_effect_medical[relation]
            ):
                pg.update()
                pg.display()
                continue
            attempts = 0
            while True:
                if attempts:
                    time.sleep(sleep_time)
                attempts += 1
                response = requests.get(
                    url.format(
                        query=urllib.parse.quote_plus(
                            re.sub(
                                r"\ +",
                                " ",
                                query.format(
                                    relation=relation,
                                    medical_entity=medical_entity,
                                    tail_predicates="|".join(tail_predicates),
                                    chain_predicates="|".join(chain_predicates),
                                ),
                            )
                        )
                    ),
                    headers=headers,
                )
                try:
                    response.raise_for_status()
                    break
                except requests.HTTPError as error:
                    if attempts == num_attempts:
                        raise error
            cause_medical_effect_medical[relation][medical_entity] = json.loads(
                response.content.decode()
            )["results"]["bindings"]
            with open(cause_medical_effect_medical_path, "w") as file:
                json.dump(cause_medical_effect_medical, file)
            pg.update()

    with open(cause_medical_effect_medical_path, "w") as file:
        json.dump(cause_medical_effect_medical, file, indent=2)

    for relation in relations:
        for medical_entity in list(cause_medical_effect_medical[relation].keys()):
            if medical_entity not in medical_entities:
                del cause_medical_effect_medical[relation][medical_entity]

    cause_medical = parse_json(sum(cause_medical_effect_medical["?cause"].values(), []))
    cause_medical["origin"] = sum(
        (
            [key] * len(values)
            for key, values in cause_medical_effect_medical["?cause"].items()
        ),
        [],
    )
    effect_medical = parse_json(
        sum(cause_medical_effect_medical["?effect"].values(), [])
    )
    effect_medical["origin"] = sum(
        (
            [key] * len(values)
            for key, values in cause_medical_effect_medical["?effect"].items()
        ),
        [],
    )

    cause_effect["cause_medical"] = cause_effect.cause.isin(cause_medical.cause)
    cause_effect["effect_medical"] = cause_effect.effect.isin(effect_medical.effect)
    cause_effect["evaluation"] = (
        cause_effect.cause_medical & cause_effect.effect_medical
    )

    cause_origin = cause_medical.groupby("cause")["origin"].apply(
        lambda x: "|".join(x.drop_duplicates().values)
    )
    effect_origin = effect_medical.groupby("effect")["origin"].apply(
        lambda x: "|".join(x.drop_duplicates().values)
    )

    cause_effect["cause_origin"] = cause_origin.reindex(cause_effect.cause).values
    cause_effect["effect_origin"] = effect_origin.reindex(cause_effect.effect).values

    cause_effect = cause_effect.dropna(subset=["causeLabel", "effectLabel"])

    covid_related = cause_effect.causeLabel.str.contains(
        "COVID-19 pandemic"
    ) | cause_effect.effectLabel.str.contains("COVID-19 pandemic")
    unknown = (cause_effect.causeLabel == "unknown") | (
        cause_effect.effectLabel == "unknown"
    )
    cause_effect = cause_effect.loc[~covid_related]
    cause_effect = cause_effect.loc[~unknown]

    cause_effect = cause_effect.drop(["cause", "effect"], axis=1).rename(
        {"causeLabel": "cause", "effectLabel": "effect"}, axis=1
    )
    size_before = cause_effect.shape[0]
    cause_effect = cause_effect.drop_duplicates()
    duplicates = size_before - cause_effect.shape[0]

    pattern = r"^[a-zA-z]\d+$"
    cause_effect = cause_effect.loc[
        ~cause_effect.cause.str.contains(pattern)
        & ~cause_effect.effect.str.contains(pattern)
    ]
    print(
        "\n".join(
            [
                f"covid related relations: {covid_related.sum()}",
                f"unknown relations: {unknown.sum()}",
                f"duplicate relations: {duplicates}",
                f"total relations: {cause_effect.shape[0]}",
            ]
        )
    )

    cause_effect.to_csv(os.path.join(constants.WIKIDATA_PATH, "wikidata-test.csv"))


if __name__ == "__main__":
    main()
