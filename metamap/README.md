Using metamap with the Dockerfile requires additional setup:

1. Download the main metamap archive as per the MetaMap installation instructions <https://lhncbc.nlm.nih.gov/ii/tools/MetaMap/run-locally/MetaMap.html> <https://lhncbc.nlm.nih.gov/ii/tools/MetaMap/documentation/Installation.html>
2. Download any additional vocabularies you wish to use <https://lhncbc.nlm.nih.gov/ii/tools/MetaMap/documentation/DescriptionOfDataVersions.html>
3. Create an `input` and `output` directory and copy all texts to process into the input folder; entries should be separated by new lines and each line should have the form ID|text as per the --sldiID flag form <https://lhncbc.nlm.nih.gov/ii/tools/MetaMap/Docs/MM_2016_Usage.pdf>
4. Run the command `docker-compose run --rm metamap /root/input/{input_file} --mm_data_year {data_year} --mm_data_version {data_version} --relaxed_model --sldiID --fielded_mmi_output` to run MetaMap on the input_file in `input`. `{data_year}` is something along the lines of 2021AB and `{data_version}` is one of NLM, Base or USABase, depending on the data files downloaded in step 2. 
   1. Add the `--term_processing` flag when evaluating on just phrases
   2. Use the `--restrict_to_sources SNOMEDCT_US,RXNORM` flag to restrict to SNOMEDCT and RXNORM vocabularies
5. Run `mmi_to_json.py` to convert the mmi output into a more easily readable jsonl file
