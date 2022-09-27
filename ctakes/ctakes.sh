#! /bin/bash

source /root/ctakes/api_key.txt
export umlsKey
cd apache-ctakes*
exec bin/runClinicalPipeline.sh -i ./input --xmiOut ./output "$@"