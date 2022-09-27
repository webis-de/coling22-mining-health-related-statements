import argparse
import bz2
import itertools
import json
import math
import os
import pathlib
import random
import re
import tarfile
import xml.etree.ElementTree as et
from typing import Any, Dict, Iterator, Optional

try:
    import gensim

    GENSIM = True
except ImportError:
    GENSIM = False
    pass
import health_causenet.constants
import nltk
import pytorch_lightning as pl
import spacy
import torch.utils.data
import tqdm


class SpacySingleton(object):
    _instance = None

    def __init__(self, name: str) -> None:
        self._nlp = spacy.load(name)
        self.__call__ = self._nlp.__call__

    def __new__(cls, name: str):
        if cls._instance is None:
            cls._instance = super(SpacySingleton, cls).__new__(cls)
            cls._instance.__init__(name)
        return cls._instance


class RandomMultiIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, *datasets: torch.utils.data.IterableDataset) -> None:
        super().__init__()
        self.dataset_names = tuple(dataset.__class__.__name__ for dataset in datasets)
        self.datasets = tuple(itertools.cycle(dataset) for dataset in datasets)

    def __iter__(self) -> Iterator[Dict[str, bool]]:
        while True:
            idx = random.choice(range(len(self.datasets)))
            name = self.dataset_names[idx]
            dataset = self.datasets[idx]
            sample = next(dataset)
            yield {"text": sample, "labels": name != WikipediaDataset.__name__}


class Dataset(torch.utils.data.IterableDataset):
    def __init__(self, text_type: str,) -> None:
        self.text_type = text_type
        if text_type == "noun_phrase":
            self.nlp = SpacySingleton("en_core_web_sm")
        elif text_type == "sentence":
            self.nlp = nltk.tokenize.sent_tokenize
        else:
            self.nlp = lambda x: [x]

    def process(self, text: str) -> Iterator[str]:
        if isinstance(self.nlp, SpacySingleton):
            doc = self.nlp.__call__(text)
            for np in doc.noun_chunks:
                yield np.text
        else:
            sentences = self.nlp(text)
            yield from sentences

    def get_start_end(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = self.start
            iter_end = self.end
        else:
            per_worker = int(
                math.ceil((self.end - self.start) / float(worker_info.num_workers))
            )
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
        return iter_start, iter_end


class PubMedDataset(Dataset):
    def __init__(
        self,
        text_type: str,
        dump_dir: pathlib.Path = pathlib.Path(
            health_causenet.constants.PUBMED_ABSTRACT_JSONL_DIR
        ),
        total: Optional[int] = None,
    ) -> None:
        super().__init__(text_type)
        if dump_dir.suffix == ".txt":
            self.path_list = [dump_dir]
        else:
            self.path_list = list(dump_dir.glob("*.jsonl"))
        self.pg = tqdm.tqdm(total=total) if total is not None else None

    def __iter__(self) -> Iterator[str]:
        for path in self.path_list:
            with path.open("r") as file:
                for line in file:
                    if path.suffix == ".jsonl":
                        doc = json.loads(line)
                        abstract = doc["abstract"]
                        abstract = re.sub(
                            r"\(ABSTRACT TRUNCATED AT \d+ WORDS\)", "", abstract
                        )
                        yield from self.process(abstract)
                    if path.suffix == ".txt":
                        if self.text_type == "noun_phrase":
                            yield from self.process(line)
                        else:
                            yield line
                    if self.pg is not None:
                        self.pg.update()


class PubMedCentralDataset(Dataset):
    def __init__(
        self,
        text_type: str,
        dump_dir: pathlib.Path = pathlib.Path(
            health_causenet.constants.PUBMED_CENTRAL_CORPUS_PATH
        ),
        total: Optional[int] = None,
    ) -> None:
        super().__init__(text_type)
        if dump_dir.suffix == ".txt":
            self.path_list = [dump_dir]
        else:
            self.path_list = list(dump_dir.glob("*.tar.gz"))
        self.pg = tqdm.tqdm(total=total) if total is not None else None

    def __iter__(self) -> Iterator[str]:
        for path in self.path_list:
            if path.suffix == ".txt":
                with path.open("r") as file:
                    for line in file:
                        if self.text_type == "noun_phrase":
                            yield from self.process(line)
                        else:
                            yield line
            else:
                with tarfile.open(path, "r:gz") as tar:
                    for member in tar:
                        content = []
                        if not member.name.endswith(".nxml"):
                            continue
                        file = tar.extractfile(member)
                        if file is None:
                            continue
                        xml_string = file.read()
                        root = et.fromstring(xml_string)
                        content = ""
                        abstract = root.find(".//abstract")
                        if abstract is not None:
                            content += " ".join(abstract.itertext())
                        body = root.find(".//body")
                        if body is not None:
                            if content:
                                content += " "
                            content += " ".join(body.itertext())
                        content = content.replace("\n", " ")
                        if self.pg is not None:
                            self.pg.update()
                        if self.text_type:
                            yield from self.process(content)
                        else:
                            yield content


class EncyclopediaDataset(Dataset):
    def __init__(
        self,
        text_type: str,
        dump_dir: pathlib.Path = pathlib.Path(
            health_causenet.constants.ENCYCLOPEDIA_DUMP_PATH
        ),
        total: Optional[int] = None,
    ):
        super().__init__(text_type)
        if dump_dir.suffix == ".txt":
            self.pathlist = [dump_dir]
        else:
            self.pathlist = list(dump_dir.glob("*.json"))
        self.pg = tqdm.tqdm(total=total) if total is not None else None

    def __iter__(self) -> Iterator[str]:
        for path in self.pathlist:
            with path.open("r") as file:
                if path.suffix == "json":
                    encyclopedia_entries = json.load(file)
                    for encyclopedia_entry in encyclopedia_entries:
                        entry = (
                            encyclopedia_entry["title"]
                            + ". "
                            + encyclopedia_entry["entry"]
                        )
                        entry = entry.replace("\n", " ")
                        entry = re.sub(r" +", " ", entry)
                        yield from self.process(entry)
                        if self.pg is not None:
                            self.pg.update()
                else:
                    for line in file:
                        if self.text_type == "noun_phrase":
                            yield from self.process(line)
                        else:
                            yield line


class WikipediaDataset(Dataset):
    def __init__(
        self,
        text_type: str,
        dump_dir: pathlib.Path = pathlib.Path(
            health_causenet.constants.WIKIPEDIA_DUMP_PATH
        ),
        total: Optional[int] = None,
    ) -> None:
        super().__init__(text_type)
        self.dump_dir = dump_dir
        self.pg = tqdm.tqdm(total=total) if total is not None else None

    def __iter__(self) -> Iterator[str]:
        if self.dump_dir.suffix == ".bz2":
            dump_file = bz2.BZ2File(self.dump_dir)
        else:
            dump_file = self.dump_dir.open("r")
        with dump_file as file:
            if ".xml" in self.dump_dir.suffixes:
                if not GENSIM:
                    raise ImportError("to use xml wikipedia dataset, import gensim")
                iterator = gensim.corpora.wikicorpus.extract_pages(file, ("0",), None)
                for _, text, _ in iterator:
                    text = gensim.corpora.wikicorpus.filter_wiki(text)
                    if text.startswith("#REDIRECT"):
                        continue
                    text = re.sub(r"\n+", " ", text).strip() + "\n"
                    text = re.sub(r"'+", "'", text).strip().strip("'")
                    text = re.sub(r"={2,}[a-zA-Z ]+={2,}", "", text).strip().strip("'")
                    text = re.sub(r" +", " ", text).strip()
                    yield from self.process(text)
                    if self.pg is not None:
                        self.pg.update()
            else:
                for line in file:
                    if self.text_type == "noun_phrase":
                        yield from self.process(line)
                    else:
                        yield line


class HealthBertDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        text_type: str = "sentence",
        health_corpus_type: str = "pubmed",
        wikipedia_corpus_path: Optional[str] = None,
        health_corpus_path: Optional[str] = None,
        batch_size: int = 1,
    ):
        super().__init__()
        self.text_type = text_type
        self.health_corpus_type = health_corpus_type
        self.wikipedia_corpus_path = wikipedia_corpus_path
        self.health_corpus_path = health_corpus_path
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        nltk.download("punkt")

    def setup(self, stage: Optional[str] = None) -> None:
        health_corpus_kwargs = (
            {}
            if self.health_corpus_path is None
            else {"dump_dir": pathlib.Path(self.health_corpus_path)}
        )
        wikipedia_corpus_kwargs = (
            {}
            if self.wikipedia_corpus_path is None
            else {"dump_dir": pathlib.Path(self.wikipedia_corpus_path)}
        )
        wikipedia_dataset = WikipediaDataset(self.text_type, **wikipedia_corpus_kwargs)
        if self.health_corpus_type == "pubmed":
            health_dataset = PubMedDataset(self.text_type, **health_corpus_kwargs)
        elif self.health_corpus_type == "encyclopedia":
            health_dataset = EncyclopediaDataset(self.text_type, **health_corpus_kwargs)
        else:
            raise ValueError(
                f"invalid value for health_corpus_type, "
                f"expected on of [pubmed, encyclopedia], got f{self.health_corpus_type}"
            )
        self.dataset = RandomMultiIterableDataset(wikipedia_dataset, health_dataset)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "type",
        type=str,
        choices=("wikipedia", "pubmed", "encyclopedia", "pubmed_central"),
    )
    parser.add_argument("out_dir", type=str)
    parser.add_argument(
        "--text_type", type=str, default=None, choices=("sentence", "noun_phrase")
    )
    parser.add_argument("--path", type=str, default=None)

    args = parser.parse_args()

    iterator = None
    total = health_causenet.constants.DOC_COUNTS[args.type]
    kwargs: Dict[str, Any] = {"total": total}
    if args.path is not None:
        kwargs["dump_dir"] = pathlib.Path(args.path)
    if args.type == "wikipedia":
        iterator = WikipediaDataset(args.text_type, **kwargs)
    if args.type == "pubmed":
        iterator = PubMedDataset(args.text_type, **kwargs)
    if args.type == "encyclopedia":
        iterator = EncyclopediaDataset(args.text_type, **kwargs)
    if args.type == "pubmed_central":
        iterator = PubMedCentralDataset(args.text_type, **kwargs)

    assert iterator is not None

    out_path = os.path.join(args.out_dir, f"{args.type}.txt")
    with open(out_path, "w") as file:
        for text in iterator:
            if args.text_type == "sentence":
                text = text.strip()
            if len(text) < 10:
                continue
            file.write(text + "\n")


if __name__ == "__main__":
    main()
