import pubmed_parser

DIRECTORY = "/mnt/ceph/storage/corpora/corpora-thirdparty/corpus-pubmed"


def test_from_xml():
    crawler = pubmed_parser.Crawler()
    xml = next(crawler.generator(DIRECTORY))
    article = xml.find("PubmedArticle")
    assert article is not None
    pubmed_parser.Article.from_xml(article)
