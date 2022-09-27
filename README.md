# Health CauseNet

This repository is used to evaluate several termhood scores for extracting health-related statements from web data. Python requirements are contained in the `requirements.txt` file.

Samples of the resources extracted from the CauseNet can be found in `data/` directory.

## Corpora

To fully evaluate all approaches several prerequisite datasets need to downloaded. First, a Wikipedia (https://dumps.wikimedia.org/), PubMed (https://pubmed.ncbi.nlm.nih.gov/download/, https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/), PubMed Central (https://www.ncbi.nlm.nih.gov/pmc/tools/textmining/, https://ftp.ncbi.nlm.nih.gov/pub/pmc/) and NCBI Bookshelf (https://www.ncbi.nlm.nih.gov/books/about/ftp/, https://ftp.ncbi.nlm.nih.gov/pub/litarch/) dump need to be downloaded. The Encyclopedia corpus can be scraped using the scraping scripts in the `encyclopedia` directory. All corpora download locations then need to be linked in the `health_causenet/constants.py` file.

## Termhood Scores

To use the termhood scores, n-gram corpus frequencies need to computed for each corpus. This can be done using the `health_causenet/cf.py` script. The NCBI ids of the specific textbooks are contained in the `data/textbook_ids.txt` file and should be passed as an argument to the python script when parsing the textbook corpus frequencies.

## Vocabulary Approaches

Running the vocabulary approaches first requires downloading the UMLS (https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html) and creating the Full UMLS and RxNorm + SNOMEDCT subsets. We refer to the instruction manual of the Metamorphosys complete instructions of how to setup this step (https://www.ncbi.nlm.nih.gov/books/NBK9683/). After creating both vocabularies, their paths also need to linked to in the `health_causenet/constants.py` file.

Detailed instructions for how to run cTakes and MetaMap can be found in the `ctakes/` and `metamap/` directories.

Enabling the custom vocabularies to be used with QuickUMLS requires running the quickumls install script `python -m quickumls.install {PATH_TO_UMLS_DIR} {TARGET_DIR}`. `create_quick_umls_dbs.sh` is an example script to creating the necessary files. The location of the QuickUMLS databases needs to be linked using `health_causenet/constants.py` file.

Enabling the custom vocabularies to be used with ScispaCy requires running the `create_scispacy_umls_kb.py` script. It takes the directory containing the META UMLS directory and the target directory to save the ScispaCy vectors as parameters. The location of the ScispaCy vectors needs to be linked using the `health_causenet/constants.py` file.

## BERT Approaches

Scripts to fine-tune BERT models on the health-related phrase detection task can be found in the `health_bert` directory. First, the corpora need to be preprocessed using `python -m health_bert.data` script. It takes a Wikipedia, PubMed or Encyclopedia corpus path as input and writes the sentences or noun phrases to a specified text file. The text files can then be passed as argument to the `health_bert/train.py` script to fine-tune pretrained BERT models. Call `python -m health_bert.train -h` for a list of parameters to adjust training. An example script for fine-tuning on a slurm cluster is contained in `sbatch-train-health-bert.sh`.

The main functionality is located in the `health_causenet/causenet.py` file. Further supplementary code is found in the `encyclopedia` directory, which can be used to scrape encyclopedia entries and the `health_bert` directory, which contains code for training and running a health-phrase detection BERT model. Analysis of the approaches can be done using the `exploration.ipynb` notebook.

## Evaluation Datasets

Creating the CauseNet evaluation datasets requires downloading the CauseNet https://causenet.org/. The CauseNet files need to processed into parquet files using the `causenet_to_parquet.py` script. Next, use the `wikidata.py` script to parse all the Wikidata dataset. Then link the download location appropriately using the `health_causenet/constants.py` file. To create the evaluation datasets and run the full parameter search for each approach run all cells in the `notebooks/classify_test_causenet.ipynb` notebook. Manual evaluations for each dataset are contained in the `data/manual_evaluations.json` file.

To run the full evaluation as done in the paper, run all cells in the `notebooks/termhood_exploration.ipynb` and `notebooks/evaluate_test_causenet.ipynb` notebooks.

## Resource Creation

Creation of the web-scale resources can be done using the `extract_medical.py` script. It receives the path to CauseNet parquet files and the desired parameterization of the termhood scores as parameters. Descriptive statistics of the resources can be created using the `notebooks/evalute_full_causenet.ipynb` notebook.
