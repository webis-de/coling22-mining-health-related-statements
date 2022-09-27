CORPUS_TYPE="pubmed"
# CORPUS_TYPE="encyclopedia"

TEXT_TYPE="noun_phrase"
# TEXT_TYPE="sentence"

BERT_MODEL="bert"
# BERT_MODEL="pubmedbert"
# BERT_MODEL="scibert"
MODEL_ROOT_DIR="./models"

DATA_PATH="./data" # data path needs to be set

if [ $BERT_MODEL == "bert" ]; then
	BASE_MODEL="bert-base-uncased"
elif [ $BERT_MODEL == "pubmedbert" ]; then
	BASE_MODEL="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
elif [ $BERT_MODEL == "scibert" ]; then
	BASE_MODEL="allenai/scibert_scivocab_uncased"
else
	echo "invalid bert base model"
	exit
fi

python -m health_bert.train \
	--gpus -1 \
	--batch_size 16 \
	--text_type ${TEXT_TYPE} \
	--health_corpus_type ${CORPUS_TYPE} \
	--bert_base_name ${BASE_MODEL} \
	--accumulate_grad_batches 2 \
	--default_root_dir ${MODEL_ROOT_DIR}/${BERT_MODEL}_${TEXT_TYPE}_${CORPUS_TYPE} \
	--monitor loss \
	--lr 0.00001 \
	--seed 42 \
	--every_n_train_steps 1000 \
	--patience 15 \
	--health_corpus_path /train_data/${CORPUS_TYPE}.txt \
	--wikipedia_corpus_path /train_data/wikipedia.txt \
	--save_top_k 2
