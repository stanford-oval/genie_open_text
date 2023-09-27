# Wiki-LLM

## Setup

Clone the repo:

```
git clone https://github.com/stanford-oval/wiki_llm.git
cd wiki_llm
```

Create & activate env:

```
conda env create --file conda_env.yaml
conda activate wiki_llm
python -m spacy download en_core_web_sm
```


Set up GPT-3/GPT-4:
First, find the secret OpenAI API key from [this](https://beta.openai.com/account/api-keys) page.
Then, create a text file named `API_KEYS` in this folder and put your OpenAI API key for access to GPT-3 models:
`export OPENAI_API_KEY=<your OpenAI API key>`.
Also specify the type of API you are using: `export OPENAI_API_TYPE=<azure or openai>`.
This file is in `.gitignore` since it is important that you do not accidentally commit your key.
Set up Azure Cosmos DB for MongoDB (for back-end):
In the same file `API_KEYS`, add `export COSMOS_CONNECTION_STRING=<your COSMOS connection string>`


## Bring up ColBERT

To avoid waiting long for the ColBERT index to be loaded, run the Flask script below that brings up ColBERT and accepts HTTP requests.

```
make start-colbert
```

Example curl command:

```
curl http://127.0.0.1:5000/search -d '{"query": "who is the current monarch of the united kingdom?", "evi_num": 5}' -X GET -H 'Content-Type: application/json'
```

## Interactive Chat via Command Line

You can run different pipelines using commands like these:

```
make demo pipeline=early_combine engine=gpt-4
make demo pipeline=verify_and_correct
make demo pipeline=retrieve_and_generate temperature=0
```

## Interactive Chat via a Web Interface

Talk to the chatbot with our correction methods in a web interface.

### Bring up backend API

```
make start-backend
```

Example curl command to test the backend:

```
curl http://127.0.0.1:5001/chat -d '{"experiment_id": "test_experiment", "dialog_id": "test_dialog", "turn_id": 0, "system_name": "retrieve_and_generate", "new_user_utterance": "who stars in the wandering earth 2?"}' -X POST -H 'Content-Type: application/json'
```

The front-end is located at https://github.com/stanford-oval/wikichat, which you can deploy separately on another VM or via vercel.com


## ColBERT Indexing
Update the following variables in `Makefile` if need:
```
##### ColBERT indexing #####
nbits ?= 2# encode each dimension with 2 bits
doc_maxlen ?= 160# number of "tokens", to account for (120 "words" + title) that we include in each wikipedia paragraph
checkpoint ?= ./workdir/colbert_model
split ?= all# '1m' passages or 'all'
experiment_name ?= wikipedia_$(split)
index_name ?= wikipedia.$(split).$(nbits)bits
wiki_date ?= 03_23_2023
collection ?= ./workdir/wikipedia_$(wiki_date)/collection_$(split).tsv
nranks ?= 4# number of GPUs to use
```
Once you've updated these variables, run the following command to start indexing:

Download latest english wikipedia dump:
```
make download-latest-wiki
```

Run wikiextractor:
```
make extract-wiki
```
This will extract the pages into a set of sharded files, which will be located in the text directory. This process takes a few hours and can be potentially made faster with more threads. 

Run
```
make split-wiki
```
This script will split the Wikipedia documents into blocks, with each block containing up to `MAX_BLOCK_WORDS` words. It will write these blocks into the .tsv file `$(collection)` which is required for ColBERT indexing.

Finally, run this command to start ColBERT indexing:

```
make index-wiki
```