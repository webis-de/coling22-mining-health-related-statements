import argparse
import bz2
import json
import pathlib
import re
import tarfile
import xml.etree.ElementTree as et
from typing import Generator, Optional

import gensim
import pandas as pd
import tqdm

import health_causenet.corpus
import health_causenet.constants
import pubmed_parser.article
import pubmed_parser.xml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "corpus",
        type=str,
        choices=[
            "pubmed",
            "pubmed_central",
            "textbook",
            "encyclopedia",
            "umls",
            "encyclopedia_umls",
            "wikipedia",
        ],
        help="corpus name for counting frequencies",
    )
    parser.add_argument(
        "--dump_dir", type=str, default="", help="directory of corpus dump",
    )
    parser.add_argument(
        "--textbook_id_path",
        type=str,
        default=health_causenet.constants.TEXTBOOK_IDS_PATH,
        help="path to textbook ids file",
    )
    parser.add_argument(
        "-P",
        "--num_processes",
        type=int,
        default=1,
        help="number of parallel processes",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1000000,
        help="maximum number of tokens in vocabulary",
    )
    parser.add_argument(
        "--max_ram", type=int, default=95, help="maximum ram usage before pruning"
    )
    parser.add_argument(
        "--prune_threshold",
        type=float,
        default=2,
        help="factor of too many tokens before pruning",
    )
    parser.add_argument(
        "--checkpoint_freq", type=float, default=0.1, help="checkpoint frequency",
    )
    parser.add_argument(
        "--n_gram_size", nargs=2, type=int, default=(1, 1), help="n gram window size"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=health_causenet.constants.BASE_PATH,
        help="path to save parquet cf df file",
    )

    args = parser.parse_args()

    return args


def pubmed_docs(dump_dir: pathlib.Path) -> Generator[str, None, None]:
    def article_processor(article: pubmed_parser.article.Article) -> Optional[str]:
        if article.language == "eng":
            return article.title + "\n" + article.abstract
        return ""

    article_generator = pubmed_parser.xml.ArticleGenerator(
        article_processor=article_processor
    )

    for content in article_generator(*[str(path) for path in dump_dir.glob("*")]):
        yield content


def pubmed_central_docs(dump_dir: pathlib.Path) -> Generator[str, None, None]:
    for tar_path in dump_dir.glob("*.tar.gz"):
        tar = tarfile.open(tar_path, "r:gz")
        for member in tar:
            content = []
            if not member.name.endswith(".nxml"):
                continue
            file = tar.extractfile(member)
            if file is None:
                continue
            xml_string = file.read()
            root = et.fromstring(xml_string)
            iterator = root.findall(".//abstract//p") + root.findall(".//body//p")
            for paragraph in iterator:
                if paragraph.text is not None:
                    content.append(paragraph.text.rstrip(" \n.") + ".")
                if paragraph.text is not None:
                    content.append(paragraph.text.rstrip(" \n.") + ".")
            yield " ".join(content)


def textbook_docs(
    dump_dir: pathlib.Path, textbook_id_path: pathlib.Path
) -> Generator[str, None, None]:
    with open(textbook_id_path, "r") as file:
        ids = file.read().splitlines()
    books_index = pd.read_csv(dump_dir.joinpath("file_list.csv"), index_col=False)

    book_paths = books_index.set_index("Accession ID").loc[ids]["File"]
    book_paths = book_paths.map(lambda x: "/".join(x.split("/")[1:]))
    book_paths = book_paths.map(lambda x: dump_dir.joinpath(x))
    book_paths = book_paths.values.tolist()
    for path in book_paths:
        tar = tarfile.open(path, "r:gz")
        content = ""
        for member in tar.getmembers():
            if not member.name.endswith(".nxml"):
                continue
            if member.name.endswith("TOC.nxml"):
                continue
            file = tar.extractfile(member)
            root = et.fromstring(file.read())
            for paragraph in root.iter("p"):
                content += paragraph.text if paragraph.text is not None else ""
                content += paragraph.tail if paragraph.tail is not None else ""
                sub_content_iterator = paragraph.iter()
                next(sub_content_iterator)  # remove paragraph root
                for sub_content in sub_content_iterator:
                    content += sub_content.text if sub_content.text is not None else ""
                    content += sub_content.tail if sub_content.tail is not None else ""
        yield content


def wikipedia_docs(dump_dir: pathlib.Path) -> Generator[str, None, None]:
    if dump_dir.suffix == ".bz2":
        dump_file = bz2.BZ2File(dump_dir)
    else:
        dump_file = open(dump_dir)
    with dump_file as file:
        iterator = gensim.corpora.wikicorpus.extract_pages(file, ("0",), None)
        for _, text, _ in iterator:
            text = gensim.corpora.wikicorpus.filter_wiki(text)
            if text.startswith("#REDIRECT"):
                continue
            text = re.sub(r"\n+", " ", text).strip() + "\n"
            text = re.sub(r"'+", "'", text).strip().strip("'")
            text = re.sub(r"={2,}[a-zA-Z ]+={2,}", "", text).strip().strip("'")
            text = re.sub(r" +", " ", text).strip()
            yield text
    file.close()


def encyclopedia_docs(dump_dir: pathlib.Path) -> Generator[str, None, None]:
    for json_path in dump_dir.glob("*.json"):
        with open(json_path, "r") as file:
            encyclopedia_entries = json.load(file)
        for encyclopedia_entry in encyclopedia_entries:
            yield encyclopedia_entry["title"] + "\n" + encyclopedia_entry["entry"]


def umls_docs(dump_dir: pathlib.Path) -> Generator[str, None, None]:
    definitions_file = dump_dir.joinpath("MRDEF.RRF")
    with open(definitions_file, "r") as file:
        for line in file:
            definition = line.split("|")[5]
            yield definition


def encyclopedia_umls_docs(
    umls_dump_dir: pathlib.Path, encyclopedia_dump_dir: pathlib.Path
) -> Generator[str, None, None]:
    for json_path in encyclopedia_dump_dir.glob("*.json"):
        with open(json_path, "r") as file:
            encyclopedia_entries = json.load(file)
        for encyclopedia_entry in encyclopedia_entries:
            yield encyclopedia_entry["title"] + "\n" + encyclopedia_entry["entry"]
    definitions_file = umls_dump_dir.joinpath("MRDEF.RRF")
    with open(definitions_file, "r") as file:
        for line in file:
            definition = line.split("|")[5]
            yield definition


def main():

    args = parse_args()

    pg = tqdm.tqdm(total=health_causenet.constants.DOC_COUNTS[args.corpus])

    corpus = health_causenet.corpus.Corpus(
        n_grams=args.n_gram_size, max_tokens=args.max_tokens
    )

    default_dump_dirs = {
        "pubmed": health_causenet.constants.PUBMED_ABSTRACT_XML_DIR,
        "pubmed_central": health_causenet.constants.PUBMED_CENTRAL_CORPUS_PATH,
        "textbook": health_causenet.constants.TEXTBOOK_CORPUS_PATH,
        "wikipedia": health_causenet.constants.WIKIPEDIA_DUMP_PATH,
        "umls": health_causenet.constants.UMLS_FULL_PATH,
        "encyclopedia": health_causenet.constants.ENCYCLOPEDIA_DUMP_PATH,
    }
    if args.dump_dir == "":
        dump_dir = pathlib.Path(default_dump_dirs[args.corpus])
    else:
        dump_dir = pathlib.Path(args.dump_dir)

    if args.corpus == "pubmed":
        docs = pubmed_docs(dump_dir)
    elif args.corpus == "pubmed_central":
        docs = pubmed_central_docs(dump_dir)
    elif args.corpus == "textbook":
        textbook_id_path = pathlib.Path(args.textbook_id_path)
        docs = textbook_docs(dump_dir, textbook_id_path)
    elif args.corpus == "wikipedia":
        docs = wikipedia_docs(dump_dir)
    elif args.corpus == "encyclopedia":
        docs = encyclopedia_docs(dump_dir)
    else:
        raise ValueError(
            f"invalid option for corpus, expected one of [pubmed, "
            f"pubmed_central, textbook, encyclopedia, wikipedia] got {args.corpus}"
        )

    output_path = pathlib.Path(args.output_path)
    checkpoint_path = output_path.joinpath(f"checkpoint-{args.corpus}")
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    checkpoint_start = 0
    for checkpoint in checkpoint_path.glob("*.json"):
        checkpoint_end = int(checkpoint.stem)
        if checkpoint_end > checkpoint_start:
            checkpoint_start = checkpoint_end
    if checkpoint_start:
        with open(checkpoint_path.joinpath(f"{checkpoint_start}.json"), "r") as file:
            cfs = json.load(file)
            corpus.cfs = cfs

    corpus.count_documents(
        docs,
        pg,
        strip="' -/",
        max_ram=args.max_ram,
        prune_threshold=args.prune_threshold,
        checkpoint_path=checkpoint_path,
        checkpoint_freq=args.checkpoint_freq,
        checkpoint_start=checkpoint_start,
    )

    cfs = {}
    for _cfs in corpus.cfs.values():
        cfs = {**cfs, **_cfs}

    cf_series = pd.Series(cfs)
    cf_series.name = "corpus_frequency"
    cf_df = cf_series.to_frame()
    cf_df.index.name = "term"

    cf_df.to_parquet(str(output_path.joinpath(f"{args.corpus}_cf.parquet")))


if __name__ == "__main__":
    main()
