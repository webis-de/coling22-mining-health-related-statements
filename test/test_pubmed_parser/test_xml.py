import xml.etree.ElementTree as et

# from pubmed_parser import MedlineXMLTreeBuilder, ArticleGenerator

XML_FILE = (
    "/mnt/ceph/storage/corpora/corpora-thirdparty/corpus-pubmed/xml/pubmed21n0001.xml"
)

XML_FILES = (
    "/mnt/ceph/storage/corpora/corpora-thirdparty/corpus-pubmed/xml/pubmed21n0001.xml",
    "/mnt/ceph/storage/corpora/corpora-thirdparty/corpus-pubmed/xml/pubmed21n0002.xml",
)


def test_xml():
    builder = MedlineXMLTreeBuilder()
    parser = et.XMLParser(target=builder)
    xml = et.parse(XML_FILE, parser)


def test_article_generator():
    article_generator = ArticleGenerator()
    num_articles = 0
    for _ in article_generator(*XML_FILES):
        num_articles += 1
    assert num_articles == 60000
