BASE_PATH = (
    "/mnt/ceph/storage/data-in-progress/data-research/"
    "web-search/COLING-22/coling22-health-causenet/"
)
CORPORA_PATH = "/mnt/ceph/storage/corpora/corpora-thirdparty/"
PUBMED_CORPUS_PATH = CORPORA_PATH + "corpus-pubmed/"
PUBMED_CENTRAL_CORPUS_PATH = CORPORA_PATH + "corpus-pubmed-central/"
TEXTBOOK_CORPUS_PATH = CORPORA_PATH + "corpus-pubmed-litarch/"
WIKIPEDIA_CORPUS_PATH = CORPORA_PATH + "corpus-wikipedia/"
ENCYCLOPEDIA_DUMP_PATH = BASE_PATH + "encyclopedias/"
CAUSENET_JSON_PATH = (
    BASE_PATH + "causenet-data/causality-graphs/integration/causenet-full.jsonl.bz2"
)
WIKIDATA_PATH = BASE_PATH + "wikidata"
CAUSENET_PARQUET_PATH = BASE_PATH + "causenet"
QUICKUMLS_FULL_PATH = BASE_PATH + "QuickUMLS/full"
QUICKUMLS_RX_SNO_PATH = BASE_PATH + "QuickUMLS/rx_sno"
SCISPACY_FULL_PATH = BASE_PATH + "scispacy/full"
SCISPACY_RX_SNO_PATH = BASE_PATH + "scispacy/rx_sno"
PUBMED_ABSTRACT_XML_DIR = PUBMED_CORPUS_PATH + "xml/"
PUBMED_ABSTRACT_PARQUET_DIR = PUBMED_CORPUS_PATH + "parquet/"
PUBMED_ABSTRACT_JSONL_DIR = PUBMED_CORPUS_PATH + "jsonl/"
WIKIPEDIA_DUMP_PATH = (
    WIKIPEDIA_CORPUS_PATH
    + "wikimedia-snapshots/enwiki-20210601/enwiki-20210601-pages-articles.xml.bz2"
)
CF_PATH = BASE_PATH + "cf/"
PUBMED_CF_PATH = CF_PATH + "pubmed_cf.parquet"
PUBMED_CENTRAL_CF_PATH = CF_PATH + "pubmed_central_cf.parquet"
TEXTBOOK_CF_PATH = CF_PATH + "textbook_cf.parquet"
WIKIPEDIA_CF_PATH = CF_PATH + "wikipedia_cf.parquet"
ENCYCLOPEDIA_CF_PATH = CF_PATH + "encyclopedia_cf.parquet"
UMLS_CF_PATH = CF_PATH + "umls_cf.parquet"
ENCYCLOPEDIA_UMLS_CF_PATH = CF_PATH + "encyclopedia_umls_cf.parquet"
NUM_ABSTRACTS = 31847923
NUM_TEXTBOOKS = 434
NUM_WIKIPEDIA_ARTICLES = 12583513
NUM_PUBMED_CENTRAL_ARTICLES = 3611361
NUM_UMLS_DEFINITIONS = 291835
NUM_UCSF_HEALTH_ENTRIES = 1372
NUM_MEDLINE_PLUS_ENTIRES = 4472
NUM_HEALTH_AM_ENTIRES = 8142
NUM_RX_LIST_HEALTH_ENTIRES = 16621
NUM_MERRIAM_WEBSTER_ENTIRES = 37360
DOC_COUNTS = {
    "textbook": NUM_TEXTBOOKS,
    "wikipedia": NUM_WIKIPEDIA_ARTICLES,
    "pubmed_central": NUM_PUBMED_CENTRAL_ARTICLES,
    "pubmed": NUM_ABSTRACTS,
    "umls": NUM_UMLS_DEFINITIONS,
    "encyclopedia_umls": (
        NUM_UMLS_DEFINITIONS
        + NUM_UCSF_HEALTH_ENTRIES
        + NUM_MEDLINE_PLUS_ENTIRES
        + NUM_HEALTH_AM_ENTIRES
        + NUM_RX_LIST_HEALTH_ENTIRES
        + NUM_MERRIAM_WEBSTER_ENTIRES
    ),
    "encyclopedia": (
        NUM_UCSF_HEALTH_ENTRIES
        + NUM_MEDLINE_PLUS_ENTIRES
        + NUM_HEALTH_AM_ENTIRES
        + NUM_RX_LIST_HEALTH_ENTIRES
        + NUM_MERRIAM_WEBSTER_ENTIRES
    ),
}
MANUAL_EVALUATION_PATH = BASE_PATH + "manual_evaluation.json"
TEXTBOOK_IDS_PATH = BASE_PATH + "textbook_ids.txt"
MEDICAL_CF_MAP = {
    "pubmed": PUBMED_CF_PATH,
    "textbook": TEXTBOOK_CF_PATH,
    "pubmed_central": PUBMED_CENTRAL_CF_PATH,
    "encyclopedia": ENCYCLOPEDIA_CF_PATH,
    "umls": UMLS_CF_PATH,
    "encyclopedia_umls": ENCYCLOPEDIA_UMLS_CF_PATH,
}
ST21PV = [
    "T005",
    "T007",
    "T017",
    "T022",
    "T031",
    "T033",
    "T037",
    "T038",
    "T058",
    "T062",
    "T074",
    "T082",
    "T091",
    "T092",
    "T097",
    "T098",
    "T103",
    "T168",
    "T170",
    "T201",
    "T204",
]
TRANSFORMERS_CACHE = BASE_PATH + "transformers/cache"
