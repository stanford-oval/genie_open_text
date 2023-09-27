# Processing StackExchange Corpus

This directory contains code for processing StackExchange raw corpus into tab-separated file format.

## Get Started

1. Select a domain of interest from https://stackexchange.com/sites and download its corresponding corpus
   from https://archive.org/details/stackexchange.
2. Unzip the .7z file. You may use a decompression software you have or the helper script [unzip_7z.py](unzip_7z.py).
3. Process xml file. This part is credited to [PyStack](https://github.com/zhenv5/PyStack) (minor modifications are
   applied). Under `Pystack/pre_processing`, run
    ```shell
   python pystack.py --input {unzip_file_dir} --task all 
   ```
4. Build corpus with tab-separated file format which can directly works with ColBERT. We view each question-answer pair
   as a single document. To ensure the corpus quality, we remove those answers with negative voting in the StackExchange
   community. Under this directory, run
   ```shell
   python build_corpus.py --input {unzip_file_dir} --domain {domain_name}
   ```
   The processed corpus will be located in `{unzip_file_dir}/{domain_name}_collection.tsv`.
