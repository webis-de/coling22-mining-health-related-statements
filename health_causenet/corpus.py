import datetime
import math
import json
import multiprocessing as mp
import pathlib
import re
import time
from multiprocessing.connection import Connection
from typing import Callable, Dict, Generator, List, Optional, Tuple, TypeVar

import gensim
import psutil
import tqdm
from nltk.tokenize.treebank import TreebankWordTokenizer

Input = TypeVar("Input")
Output = TypeVar("Output")


class MultiProcessor:
    def __init__(
        self,
        func: Callable[[Input], Output],
        generator: [Input, None, None],
        num_processes: int = 1,
    ) -> None:
        worker_recv, feeder_send = zip(*[mp.Pipe(False) for _ in range(num_processes)])
        collector_recv, worker_send = zip(
            *[mp.Pipe(False) for _ in range(num_processes)]
        )

        feeder_conn, collector_conn = mp.Pipe()

        self.feeder = Feeder(generator, feeder_send, collector_conn)
        self.collector = Collector(collector_recv, feeder_conn)

        self.feeder_proc = mp.Process(target=self.feeder.run)
        self.feeder_proc.start()
        self.worker_procs = [
            mp.Process(target=Worker(func, recv, send))
            for recv, send in zip(worker_recv, worker_send)
        ]

    def start(self):
        for proc in self.worker_procs:
            proc.start()

    def run(self) -> Generator[Output, None, None]:
        for out in self.collector.start():
            yield out

        self.close()

    def close(self):
        self.feeder_proc.join()
        for worker_proc in self.worker_procs:
            worker_proc.join()
            worker_proc.close()


class Worker:
    def __init__(
        self, func: Callable[[Input], Output], recv: Connection, send: Connection
    ) -> None:
        self.func = func
        self.recv = recv
        self.send = send

    def __call__(self) -> None:
        while True:
            inp = self.recv.recv()
            if inp is None:
                break
            self.send.send(self.func(inp))
        self.send.send(None)


class Feeder:
    def __init__(
        self,
        generator: Generator[Input, None, None],
        worker_sends: Tuple[Connection],
        collector_conn: Connection,
    ) -> None:
        self.generator = generator
        self.worker_sends = worker_sends
        self.collector_conn = collector_conn

    def run(self):
        for inp in self.generator:
            idx = self.collector_conn.recv()
            self.worker_sends[idx].send(inp)
        self.collector_conn.send(None)
        for worker_conn in self.worker_sends:
            worker_conn.send(None)


class Collector:
    def __init__(
        self, worker_recvs: Tuple[Connection], feeder_conn: Connection,
    ) -> None:
        self.worker_recvs = worker_recvs
        self.feeder_conn = feeder_conn

    def start(self) -> Generator[Output, None, None]:
        for worker_idx in range(len(self.worker_recvs)):
            self.feeder_conn.send(worker_idx)

        while True:
            if self.feeder_conn.poll():
                if self.feeder_conn.recv() is None:
                    break
            for worker_idx, worker_recv in enumerate(self.worker_recvs):
                if worker_recv.poll():
                    out: Output = worker_recv.recv()
                    self.feeder_conn.send(worker_idx)
                    yield out


def _deaccent(docs: Generator[str, None, None]) -> Generator[str, None, None]:
    for doc in docs:
        yield gensim.utils.deaccent(doc)


class Tokenizer:
    def __init__(self, tokenizer: TreebankWordTokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, text: str) -> List[str]:
        doc = self.tokenizer.tokenize(text)
        if isinstance(doc, str):
            doc = [doc]
        return doc

    def tokenize(self, text: str) -> List[str]:
        return self.__call__(text)


class NGramCounter:
    def __init__(
        self,
        alpha_numeric: bool,
        strip: str,
        n_grams: Tuple[int, int],
        n_gram_sep: str,
        cfs: Optional[Dict[str, Dict[str, int]]],
    ) -> None:
        self.alpha_numeric = alpha_numeric
        self.strip = strip
        self.n_grams = n_grams
        self.n_gram_sep = n_gram_sep
        self.cfs = cfs

    def __call__(self, doc: List[str]) -> Dict[str, Dict[str, int]]:
        n_gram_lengths = list(range(self.n_grams[0], self.n_grams[1] + 1))
        if self.cfs is None:
            self.cfs = {str(length): {} for length in n_gram_lengths}
        n_grams = [
            [""] * (length) for length in range(self.n_grams[0], self.n_grams[1] + 1)
        ]
        token_idx = 0
        for token in doc:
            token = token.lower()
            if self.alpha_numeric:
                token = re.sub(r"[^a-zA-Z0-9\- /']", "", token)
            if self.strip:
                token = token.strip(self.strip)
            if not re.search(r"[a-z]", token):
                continue
            token_idx += 1
            for n_gram_idx, n_gram in enumerate(n_grams):
                n_gram.append(token)
                n_gram = n_gram[1:]
                n_grams[n_gram_idx] = n_gram
                n_gram_size = n_gram_idx + 1
                if token_idx >= n_gram_size:
                    term = self.n_gram_sep.join(n_gram)
                    self.cfs[str(n_gram_size)][term] = (
                        self.cfs[str(n_gram_size)].get(term, 0) + 1
                    )
        return self.cfs

    def count(self, doc: List[str]) -> Dict[int, Dict[str, int]]:
        return self.__call__(doc)


class Corpus:
    def __init__(
        self,
        n_grams: Tuple[int, int] = (1, 1),
        n_gram_sep: str = "|",
        max_tokens: int = 1000000,
    ) -> None:
        self.n_grams = n_grams
        n_gram_lengths = list(range(self.n_grams[0], self.n_grams[1] + 1))
        self.cfs = {str(length): {} for length in n_gram_lengths}
        self.n_gram_sep = n_gram_sep
        self.max_tokens = max_tokens

    def count_documents(
        self,
        texts: Generator[str, None, None],
        pg: Optional[tqdm.tqdm] = None,
        deaccent: bool = True,
        alpha_numeric: bool = True,
        strip: str = "",
        max_ram: int = 95,
        prune_threshold: float = 2,
        checkpoint_path: Optional[pathlib.Path] = None,
        checkpoint_freq: float = 0.1,
        checkpoint_start: int = 0,
    ) -> None:

        num_docs = 0
        num_words = 0
        start = time.time()

        if deaccent:
            texts = _deaccent(texts)

        tokenizer = Tokenizer(TreebankWordTokenizer())
        n_gram_counter = NGramCounter(
            alpha_numeric, strip, self.n_grams, self.n_gram_sep, self.cfs,
        )

        total = 0
        checkpoint_idcs = ()
        if pg is not None:
            total = pg.total
            if total is not None:
                checkpoint_idcs = list(
                    range(checkpoint_start, total, math.ceil(total * checkpoint_freq))
                )[1:]

        for text_idx, text in enumerate(texts):
            num_docs += 1
            if text_idx <= checkpoint_start and checkpoint_start:
                if pg is not None:
                    pg.update()
                continue

            doc = tokenizer(text)
            num_words += len(doc)
            n_gram_counter(doc)

            num_n_gram_terms = [len(values) for values in self.cfs.values()]
            memory = psutil.virtual_memory()
            too_much_ram = memory[2] >= max_ram
            too_many_tokens = any(
                num_n_grams > self.max_tokens for num_n_grams in num_n_gram_terms
            )
            prune_too_many_tokens = any(
                num_n_grams > self.max_tokens * prune_threshold
                for num_n_grams in num_n_gram_terms
            )
            if (too_much_ram and too_many_tokens) or prune_too_many_tokens:
                self.prune()

            if text_idx in checkpoint_idcs:
                self.checkpoint(checkpoint_path, text_idx)

            if pg is not None:
                pg.update()
                pg.set_description(f"{sum(num_n_gram_terms)}")

        self.prune()
        self.checkpoint(checkpoint_path, total)

        elapsed = time.time() - start

        print()
        print(
            f"processed {num_docs} docs and {num_words} words "
            f"in {datetime.timedelta(seconds=elapsed)}"
        )
        print()

    def checkpoint(self, checkpoint_path, end_idx):
        file_name = checkpoint_path.joinpath(f"{end_idx}.json")
        with open(file_name, "w") as file:
            json.dump(self.cfs, file)

    def prune(self):
        bad_keys_list = []
        for cfs in self.cfs.values():
            if len(cfs) > self.max_tokens:
                sorted_keys = sorted(cfs.keys(), key=cfs.get)[::-1]
                bad_keys_list.append(sorted_keys[self.max_tokens :])
            else:
                bad_keys_list.append([])

        for n_gram_size, bad_keys in zip(self.cfs.keys(), bad_keys_list):
            for bad_key in bad_keys:
                del self.cfs[n_gram_size][bad_key]
