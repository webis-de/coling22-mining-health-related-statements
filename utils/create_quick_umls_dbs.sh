#!/bin/bash

BASE_PATH="/mnt/ceph/storage/data-in-progress/data-research/web-search/health-question-answering/"
UMLS_PATH=$BASE_PATH"UMLS/"
UMLS_FULL_PATH=$UMLS_PATH"2021AB_full/2021AB/META"
UMLS_RX_SNO_PATH=$UMLS_PATH"2021AB_rx_sno/2021AB/META"
QUICKUMLS_FULL_PATH=$BASE_PATH"QuickUMLS/2021AB_full"
QUICKUMLS_RX_SNO_PATH=$BASE_PATH"QuickUMLS/2021AB_rx_sno"

python -m quickumls.install $UMLS_FULL_PATH $QUICKUMLS_FULL_PATH -L
python -m quickumls.install $UMLS_RX_SNO_PATH $QUICKUMLS_RX_SNO_PATH -L