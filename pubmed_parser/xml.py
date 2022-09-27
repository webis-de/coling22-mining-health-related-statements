import time
import multiprocessing
import queue
import xml.etree.ElementTree as et
from typing import Any, Callable, Iterator, List, Optional

import tqdm

from .article import Article


class ArticleGenerator:
    def __init__(
        self,
        xml_paths: List[str],
        max_size=1000,
        article_processor: Optional[Callable[[Article], Any]] = None,
        num_articles: Optional[int] = None,
    ) -> None:
        self._xml_paths = xml_paths
        self._parser = MedlineElementParser(self._article_handler, self._done_handler)
        self._queue = multiprocessing.Queue(max_size)
        self._article_processor = article_processor
        self._final = False
        self._pg = None
        if num_articles is not None:
            self._pg = tqdm.tqdm(total=num_articles)

    def _article_handler(self, article: Article):
        out = article
        if self._article_processor:
            out = self._article_processor(article)
        self._queue.put(out, block=True, timeout=None)

    def _done_handler(self):
        if self._final:
            self._queue.put(None)

    def _parse(self, xml_paths: List[str]):
        for idx, xml_path in enumerate(xml_paths):
            if idx == len(xml_paths) - 1:
                self._final = True
            for event, elem in et.iterparse(xml_path, events=("start", "end")):
                if event == "start":
                    self._parser.start(elem)
                elif event == "end":
                    self._parser.end(elem)

    def __iter__(self) -> Iterator[Any]:
        process = multiprocessing.Process(target=self._parse, args=(self._xml_paths,))
        process.start()
        while True:
            try:
                element: Optional[Article] = self._queue.get_nowait()
            except queue.Empty:
                time.sleep(0.001)
                continue
            if element is None:
                break
            if self._pg is not None:
                self._pg.update()
            yield element
        process.join()


class XMLParser:
    def __init__(
        self, article_handler: Callable[[Article], None] = lambda _: None
    ) -> None:
        self.article_handler = article_handler
        self.parser = MedlineElementParser(article_handler)

    def parse(self, xml_path: str) -> None:
        for event, elem in et.iterparse(xml_path, events=("start", "end")):
            if event == "start":
                self.parser.start(elem)
            elif event == "end":
                self.parser.end(elem)


class MedlineElementParser:

    BASE_TAG = "/PubmedArticleSet/PubmedArticle/"
    MEDLINE_CITATION = BASE_TAG + "MedlineCitation/"
    ARTICLE = MEDLINE_CITATION + "Article/"
    MESH_HEADING = MEDLINE_CITATION + "MeshHeadingList/MeshHeading/"

    XML_ARTICLE_KEY_MAP = {
        MEDLINE_CITATION + "PMID": "pm_id",
        MEDLINE_CITATION + "DateCompleted/Year": "year",
        MEDLINE_CITATION + "DateCompleted/Month": "month",
        MEDLINE_CITATION + "DateCompleted/Day": "day",
        ARTICLE + "ArticleDate/Year": "year",
        ARTICLE + "ArticleDate/Month": "month",
        ARTICLE + "ArticleDate/Day": "day",
        ARTICLE + "Journal/Title": "journal",
        ARTICLE + "ArticleTitle": "title",
        ARTICLE + "Abstract/AbstractText": "abstract",
        ARTICLE + "AuthorList/Author/LastName": "last_name",
        ARTICLE + "AuthorList/Author/ForeName": "first_name",
        # ARTICLE + "AuthorList/Author/CollectiveName": "authors",
        # ARTICLE + "AuthorList/Author/LastName": "authors",
        # ARTICLE + "AuthorList/Author/Suffix": "authors",
        ARTICLE + "Language": "language",
        ARTICLE + "PublicationTypeList/PublicationType": "publication_types",
        MESH_HEADING + "DescriptorName": "mesh_headings",
        # MESH_HEADING + "QualifierName": "qualifiers",
    }

    XML_ARTICLE_ATTR_MAP = {
        ARTICLE
        + "Abstract/AbstractText": {"Label": "abstract", "NlmCategory": "abstract"}
    }

    def __init__(
        self,
        article_handler: Callable[[Article], None] = lambda _: None,
        done_handler: Callable[[], None] = lambda: None,
    ):
        self._tag = ""
        self._article = Article()
        self._article_handler = article_handler
        self._done_handler = done_handler

    def start(self, element: et.Element) -> None:
        self._tag += "/" + element.tag

    def end(self, element: et.Element) -> None:
        tag = element.tag
        if tag == "PubmedArticle":
            self._article_handler(self._article)
            self._article = Article()
        if tag == "PubmedArticleSet":
            self._done_handler()
        attrs = element.attrib
        if self._tag in self.XML_ARTICLE_ATTR_MAP and attrs:
            if "NlmCategory" in attrs and attrs["NlmCategory"] == "UNASSIGNED":
                del attrs["NlmCategory"]
            if "Label" in attrs and "NlmCategory" in attrs:
                del attrs["Label"]
            attr_keys = set(self.XML_ARTICLE_ATTR_MAP[self._tag].keys()).intersection(
                attrs.keys()
            )
            for attr_key in attr_keys:
                attr_value = attrs[attr_key]
                update_key = self.XML_ARTICLE_ATTR_MAP[self._tag][attr_key]
                self._article.update(update_key, attr_value.title() + ":")
        if self._tag in self.XML_ARTICLE_KEY_MAP:
            text = "".join(element.itertext()).strip()
            self._article.update(self.XML_ARTICLE_KEY_MAP[self._tag], text)
        self._tag = self._tag[: -(len(tag) + 1)]
