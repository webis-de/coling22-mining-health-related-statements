# cTakes

Using ctakes with the Dockerfile requires additional setup:

1. Download the newest version of apache-ctakes and the ctakes resources per the installation instructions: <https://ctakes.apache.org/downloads.html>
2. Copy your UMLS api-key into an api_key.txt file in the form of `umlsKey={api_key}`
3. (Optional but recommended) Create a custom UMLS dictionary to use in ctakes

    a. Download your desired version of the UMLS (2020AB used in this work)

    b. symlink the folder to this directory under the UMLS folder, e.g. for 2020AB version the following directory structure should exist `./UMLS/2020AB`

    c. Run `docker-compose run --rm ctakes /bin/bash` and in the container, run apache-ctakes-{version}/bin/runDictionaryCreator.sh

    d. Follow the instructions and save the custom dictionary under your desired name

4. Create an `input-dir` and `output-dir` directory
5. Copy your input text files input the `input-dir` directory, the xmi-outputs will be accessible in the `output-dir` directory
6. Run `docker-compose run --rm ctakes` or `docker-compose run --rm ctakes -l org/apache/ctakes/dictionary/lookup/fast/DictionaryName.xml` to use the custom dictionary you created in step 3, replacing DictionaryName with the name given in the dictionary creator gui
7. Run `xmi_to_json.py` to convert the xmi output in to a jsonl file