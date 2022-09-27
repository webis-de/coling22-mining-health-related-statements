import pubmed_parser

# DIRECTORY = (
#     "/mnt/ceph/storage/corpora/corpora-thirdparty/corpus-pubmed/pubmed21n0001.xml"
# )

DIRECTORY = "/media/data/pubmed_test.xml"


def test_indexer():
    indexer = pubmed_parser.Indexer(DIRECTORY, "pubmed")
    indexer.index()
