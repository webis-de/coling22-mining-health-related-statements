import bz2
import json
import os
import pathlib
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import nltk
import numpy as np
import pandas as pd
import quickumls
import spacy
import torch
from health_bert import health_bert
from scispacy.abbreviation import AbbreviationDetector
from scispacy.linking import EntityLinker
import scispacy.candidate_generation
from tqdm import tqdm

from . import constants, helpers

os.environ["ARROW_PRE_0_15_IPC_FORMAT"] = "1"
tqdm.pandas()

STOP_WORDS = set(nltk.corpus.stopwords.words("english"))


class NLP:

    _NLP = {}

    def __init__(self, lang: str, pipes: Optional[List[Tuple[str, Dict[str, Any]]]]):
        key = lang + str(pipes)
        if key not in self._NLP:
            nlp = spacy.load(lang)
            if pipes is not None:
                for pipe_name, config in pipes:
                    nlp.add_pipe(pipe_name, config=config)
            self._NLP[key] = nlp
        self.nlp = self._NLP[key]


def _quick_umls_tagger(
    umls_subset: str, jaccard_threshold: float, st21pv: bool
) -> quickumls.QuickUMLS:
    accepted_semptypes = None
    if st21pv:
        accepted_semptypes = constants.ST21PV
    umls_subset_path_map = {
        "full": constants.QUICKUMLS_FULL_PATH,
        "rx_sno": constants.QUICKUMLS_RX_SNO_PATH,
    }

    if umls_subset not in umls_subset_path_map:
        raise KeyError(f"invalid quickumls model name: {umls_subset}")

    path = umls_subset_path_map[umls_subset]

    tagger = quickumls.QuickUMLS(
        path, threshold=jaccard_threshold, accepted_semtypes=accepted_semptypes
    )

    return tagger


def register_scispacy_linker(kb_dir: pathlib.Path):
    name = kb_dir.name
    linker_paths = scispacy.candidate_generation.LinkerPaths(
        ann_index=str(kb_dir.joinpath("nmslib_index.bin")),
        tfidf_vectorizer=str(kb_dir.joinpath("tfidf_vectorizer.joblib")),
        tfidf_vectors=str(kb_dir.joinpath("tfidf_vectors_sparse.npz")),
        concept_aliases_list=str(kb_dir.joinpath("concept_aliases.json")),
    )

    kb_name = f"UMLS{''.join(part.title() for part in name.split('_'))}"
    kb = type(kb_name, (scispacy.candidate_generation.KnowledgeBase,), {},)

    def constructor(self, file_path: str = str(kb_dir.joinpath("kb.jsonl"))):
        super(kb, self).__init__(file_path)

    kb.__init__ = constructor

    scispacy.candidate_generation.DEFAULT_PATHS[name] = linker_paths
    scispacy.candidate_generation.DEFAULT_KNOWLEDGE_BASES[name] = kb


def _scispacy(
    causenet: pd.DataFrame,
    model: str,
    umls_subset: str,
    threshold: float,
    st21pv: bool = False,
    verbose: bool = True,
):
    accepted_semtypes = []
    if st21pv:
        accepted_semtypes = constants.ST21PV

    register_scispacy_linker(pathlib.Path(constants.SCISPACY_FULL_PATH))
    register_scispacy_linker(pathlib.Path(constants.SCISPACY_RX_SNO_PATH))

    nlp = NLP(
        model,
        [
            ("abbreviation_detector", {}),
            (
                "scispacy_linker",
                {
                    "resolve_abbreviations": True,
                    "linker_name": umls_subset,
                    "threshold": threshold,
                },
            ),
        ],
    ).nlp
    linker = nlp.get_pipe("scispacy_linker")

    def grab_tuis(entity, token) -> Set[str]:
        tuis = set()
        if entity is not None:
            if token.i >= entity.start and token.i <= entity.end:
                umls_entities = []
                for cui, score in entity._.kb_ents:
                    if score >= threshold:
                        umls_entities.append(linker.kb.cui_to_entity[cui])
                tuis = set(
                    tui for umls_entity in umls_entities for tui in umls_entity.types
                )
        return tuis

    def _match(relation, nlp):
        doc = nlp(relation)
        num_words = 0
        num_found = 0
        entities = iter(doc.ents)
        try:
            entity = next(entities)
        except StopIteration:
            entity = None
        for token in doc:
            tuis = grab_tuis(entity, token)
            if not tuis and str(token) in STOP_WORDS:
                continue
            num_words += 1

            if entity is not None:
                while token.i >= entity.end:
                    try:
                        entity = next(entities)
                    except StopIteration:
                        entity = None
                        break

            if st21pv:
                if tuis.intersection(accepted_semtypes):
                    num_found += 1
            elif tuis:
                num_found += 1
        if num_words == 0:
            return 0
        return num_found / num_words

    relations = pd.Series(
        pd.unique(causenet.loc[:, ["cause", "effect"]].values.ravel())
    )

    if verbose:
        func = relations.progress_apply
    else:
        func = relations.apply

    medical_fraction = func(lambda term: _match(term, nlp))
    medical_fraction = medical_fraction.rename("medical_score")
    medical_fraction.index = pd.Index(relations.values)

    cause_medical = medical_fraction.loc[causenet.cause].reset_index(drop=True)
    cause_medical.name = cause_medical.name + "-cause"
    effect_medical = medical_fraction.loc[causenet.effect].reset_index(drop=True)
    effect_medical.name = effect_medical.name + "-effect"

    medical_score = pd.concat([cause_medical, effect_medical], axis=1)

    return medical_score


def _quick_umls(
    causenet: pd.DataFrame,
    umls_subset: str,
    jaccard_threshold: float,
    st21pv: bool,
    verbose: bool = True,
) -> pd.DataFrame:
    tagger = _quick_umls_tagger(umls_subset, jaccard_threshold, st21pv)

    def _match(
        relation: str,
        tagger: quickumls.QuickUMLS,
        stop_words: Set[str],
        tokenizer: nltk.tokenize.TreebankWordTokenizer,
    ) -> int:
        ngrams = tokenizer.tokenize(relation)
        ngrams = [ngram for ngram in ngrams if ngram and ngram not in stop_words]
        num_terms = len(ngrams)
        if not num_terms:
            return 0
        num_matched_terms = 0
        cuis = set()
        matches = tagger.match(relation)
        for match_dict in matches:
            cuis.add(match_dict[0]["cui"])
            match_ngrams = tokenizer.tokenize(match_dict[0]["ngram"])
            match_ngrams = [
                ngram for ngram in match_ngrams if ngram and ngram not in stop_words
            ]
            num_matched_terms += len(match_ngrams)
        return min(1, num_matched_terms / num_terms)

    relations = pd.Series(
        pd.unique(causenet.loc[:, ["cause", "effect"]].values.ravel())
    )
    tokenizer = nltk.tokenize.TreebankWordTokenizer()

    if verbose:
        func = relations.progress_apply
    else:
        func = relations.apply

    medical_fraction = func(lambda term: _match(term, tagger, STOP_WORDS, tokenizer))
    medical_fraction = medical_fraction.rename("medical_score")
    medical_fraction.index = pd.Index(relations.values)

    cause_medical = medical_fraction.loc[causenet.cause].reset_index(drop=True)
    cause_medical.name = cause_medical.name + "-cause"
    effect_medical = medical_fraction.loc[causenet.effect].reset_index(drop=True)
    effect_medical.name = effect_medical.name + "-effect"

    medical_score = pd.concat([cause_medical, effect_medical], axis=1)

    return medical_score


def _scispacy_cuis(text: str, threshold: float = 0.9999) -> str:

    nlp = NLP(
        "en_core_sci_sm",
        [
            ("abbreviation_detector", {}),
            (
                "scispacy_linker",
                {
                    "resolve_abbreviations": True,
                    "linker_name": "umls",
                    "threshold": threshold,
                },
            ),
        ],
    ).nlp
    doc = nlp(text)
    cuis = set()
    for entity in doc.ents:
        prev_score = 0
        for cui, score in entity._.kb_ents:
            if prev_score and prev_score != score:
                break
            prev_score = score
            cuis.add(cui)
    return ",".join(cuis)


def load_cf(
    pubmed: bool = True,
    pubmed_central: bool = True,
    textbook: bool = True,
    encyclopedia: bool = True,
    umls: bool = True,
    encyclopedia_umls: bool = True,
    min_corpus_frequency: int = 0,
):
    print("loading wikipedia cf...")
    cf = pd.read_parquet(
        constants.WIKIPEDIA_CF_PATH, columns=["corpus_frequency"]
    ).add_suffix("_open_domain")
    if min_corpus_frequency:
        cf = cf.loc[cf.corpus_frequency_open_domain >= min_corpus_frequency]
    if pubmed:
        print("loading pubmed cf...")
        medical_cf = pd.read_parquet(
            constants.PUBMED_CF_PATH, columns=["corpus_frequency"]
        )
        if min_corpus_frequency:
            medical_cf = medical_cf.loc[
                medical_cf.corpus_frequency_open_domain >= min_corpus_frequency
            ]
        cf = cf.join(medical_cf.add_suffix("_pubmed"), how="outer")
    if pubmed_central:
        print("loading pubmed central cf...")
        medical_cf = pd.read_parquet(
            constants.PUBMED_CENTRAL_CF_PATH, columns=["corpus_frequency"]
        )
        if min_corpus_frequency:
            medical_cf = medical_cf.loc[
                medical_cf.corpus_frequency_open_domain >= min_corpus_frequency
            ]
        cf = cf.join(medical_cf.add_suffix("_pubmed_central"), how="outer")
    if textbook:
        print("loading textbook cf...")
        medical_cf = pd.read_parquet(
            constants.TEXTBOOK_CF_PATH, columns=["corpus_frequency"]
        )
        if min_corpus_frequency:
            medical_cf = medical_cf.loc[
                medical_cf.corpus_frequency_open_domain >= min_corpus_frequency
            ]
        cf = cf.join(medical_cf.add_suffix("_textbook"), how="outer")
    if encyclopedia:
        print("loading encyclopedia cf...")
        medical_cf = pd.read_parquet(
            constants.ENCYCLOPEDIA_CF_PATH, columns=["corpus_frequency"]
        )
        if min_corpus_frequency:
            medical_cf = medical_cf.loc[
                medical_cf.corpus_frequency_open_domain >= min_corpus_frequency
            ]
        cf = cf.join(medical_cf.add_suffix("_encyclopedia"), how="outer")
    if umls:
        print("loading umls cf...")
        medical_cf = pd.read_parquet(
            constants.UMLS_CF_PATH, columns=["corpus_frequency"]
        )
        if min_corpus_frequency:
            medical_cf = medical_cf.loc[
                medical_cf.corpus_frequency_open_domain >= min_corpus_frequency
            ]
        cf = cf.join(medical_cf.add_suffix("_umls"), how="outer")
    if encyclopedia_umls:
        print("loading encyclopedia_umls cf...")
        medical_cf = pd.read_parquet(
            constants.ENCYCLOPEDIA_UMLS_CF_PATH, columns=["corpus_frequency"]
        )
        if min_corpus_frequency:
            medical_cf = medical_cf.loc[
                medical_cf.corpus_frequency_open_domain >= min_corpus_frequency
            ]
        cf = cf.join(medical_cf.add_suffix("_encyclopedia_umls"), how="outer")
    print("counting number of terms...")
    cf["num_terms"] = cf.index.str.count(r"\|") + 1
    print("done.")
    return cf


def _contrastive_score(
    causenet: pd.DataFrame,
    medical_termhood: pd.Series,
    p: float,
    n_gram_size: Tuple[int, int],
    verbose: bool = True,
) -> pd.DataFrame:
    tokenizer = nltk.tokenize.TreebankWordTokenizer()

    relations = pd.Series(
        pd.unique(causenet.loc[:, ["cause", "effect"]].values.ravel())
    )

    def p_mean(array, p) -> float:
        array = array[~np.isnan(array) & ~np.isinf(array)]
        if array.shape[0] == 0:
            return 0
        if p == float("inf"):
            return array.max()
        if p == -float("inf"):
            return array.min()
        if p == 0:
            array = array[array != 0]
            if array.shape[0] == 0:
                return 0
            return np.exp(np.log(array).sum() / array.shape[0])
        if p < 0:
            array = array[array != 0]
        if array.shape[0] == 0:
            return 0
        out = ((array ** p).sum() / array.shape[0]) ** (1 / p)
        return out

    if verbose:
        func = relations.progress_map
    else:
        func = relations.map

    relation_terms = func(tokenizer.tokenize).explode().fillna("")

    if verbose:
        func = relation_terms.groupby(level=0).progress_apply
    else:
        func = relation_terms.groupby(level=0).apply

    # rolling to create ngrams
    medical_contrastive_score = (
        func(
            lambda x: p_mean(
                np.array(
                    [
                        medical_termhood.get("|".join(terms.values), np.nan)
                        for n_gram in range(n_gram_size[0], n_gram_size[1] + 1)
                        for terms in list(x.rolling(n_gram))
                        if terms.values.shape[0] == n_gram
                    ]
                ),
                p,
            )
        )
        .rename("medical_score")
        .fillna(0)
    )
    medical_contrastive_score.index = pd.Index(relations.values)

    cause_medical = (
        medical_contrastive_score.loc[causenet.cause].reset_index(drop=True).to_frame()
    )
    effect_medical = (
        medical_contrastive_score.loc[causenet.effect].reset_index(drop=True).to_frame()
    )

    return cause_medical.join(effect_medical, lsuffix="-cause", rsuffix="-effect")


def term_domain_specificity(cf: pd.DataFrame, log: float) -> pd.Series:
    cf = cf.copy().fillna(0) + 1
    num_terms = cf.num_terms
    medical = cf.filter(regex=r".*_frequency_(?!open_domain)")
    medical_prob = np.exp(
        np.log(medical) - np.log(medical.groupby(num_terms).transform("sum"))
    )
    open_domain_prob = np.exp(
        np.log(cf.corpus_frequency_open_domain)
        - np.log(cf.corpus_frequency_open_domain.groupby(num_terms).transform("sum"))
    )
    out = np.log(medical_prob.div(open_domain_prob, axis=0) + 1) / np.log(log)
    return out.mean(axis=1)


def _term_domain_specificity(
    causenet: pd.DataFrame,
    cf: pd.DataFrame,
    log: float = np.e,
    p: float = 1,
    n_gram_size: Tuple[int, int] = (1, 1),
    verbose: bool = True,
) -> pd.DataFrame:

    medical_termhood = term_domain_specificity(cf, log)

    medical_score = _contrastive_score(
        causenet, medical_termhood, p, n_gram_size, verbose
    )

    return medical_score


def contrastive_weight(cf: pd.DataFrame, log: float, add: float) -> pd.Series:
    cf = cf.copy().fillna(0)
    medical = cf.filter(regex=r".*_frequency_(?!open_domain)")
    data_frequency = medical.add(cf.corpus_frequency_open_domain, axis=0)
    term_1 = medical
    term_2 = data_frequency.groupby(cf.num_terms).transform("sum") / data_frequency
    if log:
        term_1 = np.log(term_1 + add) / np.log(log)
        term_2 = np.log(term_2 + add) / np.log(log)
    out = (term_1 * term_2).mean(axis=1)
    out = out.replace((np.inf, -np.inf), 0)
    return out


def _contrastive_weight(
    causenet: pd.DataFrame,
    cf: pd.DataFrame,
    log: float = np.e,
    add: float = 1,
    p: float = 1,
    n_gram_size: Tuple[int, int] = (1, 1),
    verbose: bool = True,
) -> pd.DataFrame:

    medical_termhood = contrastive_weight(cf, log, add)

    medical_score = _contrastive_score(
        causenet, medical_termhood, p, n_gram_size, verbose
    )

    return medical_score


def discriminative_weight(
    cf: pd.DataFrame, contrastive_log: float, specificity_log: float,
) -> pd.Series:
    contrastive = contrastive_weight(cf, contrastive_log, contrastive_log)
    specificity = term_domain_specificity(cf, specificity_log)
    return contrastive * specificity


def _discriminative_weight(
    causenet: pd.DataFrame,
    cf: pd.DataFrame,
    contrastive_log: float = np.e,
    specificity_log: float = np.e,
    p: float = 1,
    n_gram_size: Tuple[int, int] = (1, 1),
    verbose: bool = True,
) -> pd.DataFrame:

    medical_termhood = discriminative_weight(cf, contrastive_log, specificity_log)

    medical_score = _contrastive_score(
        causenet, medical_termhood, p, n_gram_size, verbose
    )

    return medical_score


def _scispacy_cui_extraction(text: pd.Series) -> pd.Series:
    return text.map(_scispacy_cuis)


def _health_bert(
    causenet: pd.DataFrame,
    model: health_bert.HealthBert,
    verbose: bool = True,
    batch_size: int = 32,
):
    relations = pd.Series(
        pd.unique(causenet.loc[:, ["cause", "effect"]].values.ravel())
    )
    split = max(1, int(len(relations) / batch_size))
    batches = np.array_split(relations.values, split)
    medical_score = []
    if verbose:
        batches = tqdm(batches)
    with torch.no_grad():
        for batch in batches:
            out = model({"text": batch.tolist()})
            scores = torch.sigmoid(out.logits[..., 1]).numpy().tolist()
            medical_score.extend(scores)

    medical_score = pd.Series(medical_score, index=relations.values).rename(
        "medical_score"
    )

    cause_medical = medical_score.loc[causenet.cause].reset_index(drop=True).to_frame()
    effect_medical = (
        medical_score.loc[causenet.effect].reset_index(drop=True).to_frame()
    )

    return cause_medical.join(effect_medical, lsuffix="-cause", rsuffix="-effect")


def _tagger(
    causenet: pd.DataFrame, json_path: str, st21pv: bool = False, verbose: bool = True
):
    accepted_semtypes = []
    if st21pv:
        accepted_semtypes = constants.ST21PV
    tagger_data = []
    with open(json_path, "r") as file:
        for line_json in file:
            data = json.loads(line_json)
            num_found = 0
            num_words = 0
            for word, id_list in zip(data["words"], data["umls_id_list"]):
                if word in STOP_WORDS and not id_list:
                    continue
                num_words += 1
                for _, tui in id_list:
                    if st21pv and tui not in accepted_semtypes:
                        continue
                    num_found += 1
                    break
            if num_words == 0:
                medical_score = 0
            else:
                medical_score = num_found / num_words
            tagger_data.append([data["text"], medical_score])

    medical_score = pd.DataFrame(tagger_data, columns=["text", "medical_score"])
    medical_score = medical_score.set_index("text")

    cause_medical = medical_score.reindex(causenet.cause, fill_value=0).reset_index(
        drop=True
    )
    effect_medical = medical_score.reindex(causenet.effect, fill_value=0).reset_index(
        drop=True
    )

    return cause_medical.join(effect_medical, lsuffix="-cause", rsuffix="-effect")


class CauseNet:

    LINE_COUNTS = {
        "causenet-full.jsonl.bz2": 11609890,
    }

    MEDICAL_CONCEPT_EXTRACTION_FUNCS = {
        "quickumls": _quick_umls,
        "scispacy": _scispacy,
        "health_bert": _health_bert,
        "tagger": _tagger,
        "term_domain_specificity": _term_domain_specificity,
        "contrastive_weight": _contrastive_weight,
        "discriminative_weight": _discriminative_weight,
    }

    ABSTRACT_EXTRACTION_FUNCS: Dict[str, Callable[[pd.Series], pd.Series]] = {
        "scispacy_abstract_cui_extraction": _scispacy_cui_extraction
    }

    @staticmethod
    def convert_to_parquet(
        json_line_path: str, save_path: str, split_size: int = 0
    ) -> None:
        if not split_size and not save_path.endswith(".parquet"):
            raise ValueError(
                f"expected save_path to end with .parquet, got {save_path}"
            )
        file_name = os.path.split(json_line_path)[1]
        if file_name in CauseNet.LINE_COUNTS:
            line_counts = CauseNet.LINE_COUNTS[file_name]
        else:
            line_counts = helpers.count_lines(bz2.open(json_line_path, "rb"))
        data = []
        count = 0
        file_count = 0
        blank = 0
        no_support = 0
        with bz2.open(json_line_path, mode="rt") as json_line_file:
            pg = tqdm(json_line_file, total=line_counts)
            for line in pg:
                relation_data = []
                relation_dict = json.loads(line)
                relation = relation_dict["causal_relation"]
                try:
                    support = relation_dict["support"]
                    pattern_set = None
                except KeyError:
                    support = 0
                    no_support += 1
                    pattern_set = set()
                cause = relation["cause"]["concept"].replace("_", " ")
                cause = re.sub(r" +", r" ", cause)
                effect = relation["effect"]["concept"].replace("_", " ")
                effect = re.sub(r" +", r" ", effect)
                if cause == " " or effect == " " or not cause or not effect:
                    blank += 1
                    continue
                for source in relation_dict["sources"]:
                    source_type = source["type"]
                    if source_type in ("wikipedia_list", "wikipedia_infobox"):
                        continue
                    if source_type == "clueweb12_sentence":
                        reference = source["payload"]["clueweb12_page_reference"]
                    else:
                        reference = source["payload"]["wikipedia_page_title"]
                    sentence = source["payload"]["sentence"]
                    if pattern_set is not None:
                        pattern_set.add(source["payload"]["path_pattern"])
                    relation_data.append(
                        [cause, effect, source_type, reference, sentence, support]
                    )
                if not support and pattern_set:
                    for idx in range(len(relation_data)):
                        relation_data[idx][-1] = len(pattern_set)
                pg.set_description(f"{blank} blank | {no_support} no support")
                data.extend(relation_data)
                count += 1
                if split_size and count >= split_size:
                    df = pd.DataFrame(
                        data,
                        columns=[
                            "cause",
                            "effect",
                            "type",
                            "reference",
                            "sentence",
                            "support",
                        ],
                    )
                    df.to_parquet(
                        os.path.join(save_path, f"causenet_{file_count}.parquet")
                    )
                    file_count += 1
                    count = 0
                    data = []

        if split_size:
            save_path = os.path.join(save_path, f"causenet_{file_count}.parquet")
        df = pd.DataFrame(
            data,
            columns=["cause", "effect", "type", "reference", "sentence", "support"],
        )
        df.to_parquet(save_path)

    @staticmethod
    def is_medical(
        causenet: pd.DataFrame, match_func: str, num_processes: int = 1, **kwargs: Any,
    ) -> pd.DataFrame:
        func = CauseNet.MEDICAL_CONCEPT_EXTRACTION_FUNCS[match_func]
        if num_processes > 1:
            causenet = helpers.parallelize_dataframe(
                causenet, func, num_processes, **kwargs
            )
        else:
            causenet = func(causenet, **kwargs)
        return causenet

