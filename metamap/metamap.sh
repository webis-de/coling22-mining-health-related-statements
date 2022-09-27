#!/bin/bash

./bin/skrmedpostctl start
./bin/wsdserverctl start
if test -f "$1"; then
    exec cat "$1" | ./bin/metamap --silent "${@: 2}" > "/root/output/${1##*/}"
else
    exec echo "$1" | ./bin/metamap --silent "${@: 2}" > /root/output/output.txt
fi
./bin/skrmedpostctl stop
./bin/wsdserverctl stop