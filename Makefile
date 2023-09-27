### Input variables and their default values
engine ?= gpt-4#text-davinci-003#gpt-4
temperature ?= 1.0
colbert_endpoint ?= http://127.0.0.1:5000/search
local_engine_endpoint ?= http://127.0.0.1:5002
local_engine_prompt_format ?= simple# 'none', 'alpaca' or 'simple'
experiment_id ?= mturk_1# used in db-to-text to extract mturk experiments data

pipeline ?= early_combine_with_replacement# one of: verify_and_correct, retrieve_and_generate, generate, retrieve_only, early_combine, atlas
evi_num ?= 2# num evi for each subclaim verification; 3 for retrieval
reranking_method ?= voting# none or date
do_refine ?= true
skip_verification ?= false
debug_mode ?= false
refinement_prompt ?= prompts/refine.prompt
claim_prompt_template_file ?= prompts/wikipedia/split_claim_rewrite.prompt#prompts/split_claim_rewrite.prompt
summary_prompt ?= prompts/wikipedia/single_summary.prompt#prompts/single_summary.prompt
initial_search_prompt ?= prompts/wikipedia/initial_search.prompt#prompts/initial_search.prompt
evidence_combiner_prompt ?= prompts/evidence_combiner.prompt
verification_prompt ?= prompts/verify_and_correct.prompt

PIPELINE_FLAGS = --pipeline $(pipeline) \
		--engine $(engine) \
		--local_engine_endpoint $(local_engine_endpoint) \
		--local_engine_prompt_format $(local_engine_prompt_format) \
		--claim_prompt_template_file $(claim_prompt_template_file) \
		--refinement_prompt $(refinement_prompt) \
		--summary_prompt $(summary_prompt) \
		--initial_search_prompt $(initial_search_prompt) \
		--evidence_combiner_prompt $(evidence_combiner_prompt) \
		--verification_prompt $(verification_prompt) \
		--colbert_endpoint $(colbert_endpoint) \
		--reranking_method $(reranking_method) \
		--evi_num $(evi_num) \
		--temperature $(temperature)

ifeq ($(do_refine), true)
	PIPELINE_FLAGS := $(PIPELINE_FLAGS) --do_refine
else
	PIPELINE_FLAGS := $(PIPELINE_FLAGS)
endif

ifeq ($(skip_verification), true)
	PIPELINE_FLAGS := $(PIPELINE_FLAGS) --skip_verification
else
	PIPELINE_FLAGS := $(PIPELINE_FLAGS)
endif

ifeq ($(debug_mode), true)
	PIPELINE_FLAGS := $(PIPELINE_FLAGS) --debug_mode
else
	PIPELINE_FLAGS := $(PIPELINE_FLAGS)
endif

.PHONY: demo chat-batch simulate-users start-colbert start-backend download-and-index-wiki download-latest-wiki extract-wiki split-wiki index-wiki db-to-file index-index_domain_corpus start-colbert-domain demo-domain

demo:
	python chat_interactive.py \
		$(PIPELINE_FLAGS) \
		--output_file data/demo.txt \

start-colbert:
	python colbert_app.py \
		--colbert_index_path workdir/wikipedia_04_28_2023/wikipedia.all.2bits \
		--colbert_checkpoint workdir/colbert_model \
		--colbert_collection_path workdir/wikipedia_04_28_2023/collection_all.tsv

start-backend:
	python server_api.py \
		$(PIPELINE_FLAGS) \
		# --test
		# --no_logging

db-to-file:
	python db_analytics.py --experiment_id $(experiment_id) --output_file data/db_analytics/$(experiment_id).json

##### StackExchange #####
domain ?= cooking# TODO: modify this to process your domain of interest.

process-stackexchange-corpus: workdir/
	wget -O workdir/$(domain).stackexchange.com.7z https://archive.org/download/stackexchange/$(domain).stackexchange.com.7z && \
	cd stackexchange && \
	python unzip_7z.py --zip_file ../workdir/$(domain).stackexchange.com.7z --result_dir ../workdir/$(domain) && \
	cd PyStack/pre_processing && \
	python pystack.py --input ../../../workdir/$(domain) --task all && \
	cd ../.. && \
	python build_corpus.py --input ../workdir/$(domain) --domain $(domain)

##### ColBERT indexing #####
nbits ?= 2# encode each dimension with 2 bits
max_block_words ?= 100# maximum number of words in each paragraph
doc_maxlen ?= 140# number of "tokens", to account for (100 "words" + title) that we include in each wikipedia paragraph
checkpoint ?= ./workdir/colbert_model
split ?= all# '1m' passages or 'all'
experiment_name ?= wikipedia_$(split)
#index_name ?= wikipedia.$(split).$(nbits)bits
wiki_date ?= 07_20_2023
language ?= en

#collection ?= ./workdir/$(language)/wikipedia_$(wiki_date)/collection_$(split).tsv
collection ?= ./workdir/$(domain)/$(domain)_collection.tsv
#collection ?= ./stackexchange/PyStack/dataset/$(domain)/$(domain)_collection_test_index.tsv
nranks ?= 1# number of GPUs to use
#test_query ?= "What ingredients can I substitute for parsley?"
#test_query ?= "What should I do if my F-1 visa becomes out-of-date during my PhD study in the US? "
test_query ?= "When should I use unique_ptr in C++?"
index_name = $(domain).$(nbits)bits


# takes ~70 minutes. using `axel` instead of `wget` can speed up the download
download-latest-wiki: workdir/
	wget -O workdir/$(language)wiki-latest-pages-articles.xml.bz2 https://dumps.wikimedia.org/$(language)wiki/latest/$(language)wiki-latest-pages-articles.xml.bz2

# takes ~5 hours
extract-wiki: workdir/$(language)wiki-latest-pages-articles.xml.bz2
	python -m wikiextractor.wikiextractor.WikiExtractor workdir/$(language)wiki-latest-pages-articles.xml.bz2 --output workdir/$(language)/text/

# takes ~1 minute depending on the number of cpu cores and type of hard disk
split-wiki: workdir/$(language)/text/
	python wikiextractor/split_passages.py \
		--input_path workdir/$(language)/text/ \
		--output_path $(collection) \
		--max_block_words $(max_block_words)

# takes 24 hours on a 40GB A100 GPU
index-wiki: $(collection)
	python index_wiki.py \
		--nbits $(nbits) \
		--doc_maxlen $(doc_maxlen) \
		--checkpoint $(checkpoint) \
		--split $(split) \
		--experiment_name $(experiment_name) \
		--index_name $(index_name) \
		--collection $(collection) \
		--nranks $(nranks)

tar-wiki-index:
	tar -cvf colbert_wikipedia_index_$(wiki_date).tar workdir/wikipedia_$(wiki_date)/

# Below commands are newly added by Yijia
index-domain-corpus: $(collection)
	python index_domain_corpus.py \
		--nbits $(nbits) \
		--doc_maxlen $(doc_maxlen) \
		--checkpoint $(checkpoint) \
		--experiment_name $(domain) \
		--index_name $(index_name) \
		--collection $(collection) \
		--nranks $(nranks) \
		--test_query $(test_query)

start-colbert-domain:
	CUDA_VISIBLE_DEVICES="" python colbert_app.py \
		--colbert_index_path experiments/$(domain)/indexes/$(domain).2bits \
		--colbert_checkpoint workdir/colbert_model \
		--colbert_collection_path workdir/$(domain)/$(domain)_collection.tsv

demo-domain:
	python chat_interactive.py \
		$(PIPELINE_FLAGS) \
		--output_file experiments/$(domain)/$(domain)_demo.txt

start-backend-domain:
	python server_api_domain_bot.py \
		$(PIPELINE_FLAGS) \
		# --test
		# --no_logging