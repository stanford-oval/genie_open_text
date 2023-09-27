import argparse
import logging
import sys

from flask import Flask
from flask_cors import CORS
from flask_restful import Api, reqparse
import math
from functools import lru_cache
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, "./ColBERT/")

from colbert import Searcher
from colbert.infra import Run, RunConfig

app = Flask(__name__)
CORS(app)
api = Api(app)
logger = logging.getLogger(__name__)

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    "--colbert_experiment_name",
    type=str,
    default="wikipedia_all",
    help="name of colbert indexing experiment",
)
arg_parser.add_argument(
    "--colbert_index_path",
    type=str,
    default="/home/factgpt/wiki_llm/ColBERT/downloads/wikipedia_0201_2023/wikipedia.all.2bits/wikipedia.all.2bits",
    help="path to colbert index",
)
arg_parser.add_argument(
    "--colbert_checkpoint",
    type=str,
    default="/home/factgpt/wiki_llm/ColBERT/downloads/colbertv2.0",
    help="path to the folder containing the colbert model checkpoint",
)
arg_parser.add_argument(
    "--colbert_collection_path",
    type=str,
    default="/home/factgpt/wiki_llm/ColBERT/experiments/wikipedia_all/wikipedia_0201_2023/wikipedia_0201_2023/collection_all.tsv",
    help="path to colbert document collection",
)
args = arg_parser.parse_args()

req_parser = reqparse.RequestParser()
req_parser.add_argument("query", type=str, help="The search query")
req_parser.add_argument("evi_num", type=int, help="Number of documents to return")

with Run().context(RunConfig(experiment=args.colbert_experiment_name, index_root="")):
    searcher = Searcher(
        index=args.colbert_index_path,
        checkpoint=args.colbert_checkpoint,
        collection=args.colbert_collection_path,
    )


@lru_cache(maxsize=100000)
def search(query, k):
    search_results = searcher.search(
        query, k=100
    )  # retrieve more results so that our probability estimation is more accurate
    passage_ids, passage_ranks, passage_scores = search_results
    passages = [searcher.collection[passage_id] for passage_id in passage_ids]
    passage_probs = [math.exp(score) for score in passage_scores]
    passage_probs = [prob / sum(passage_probs) for prob in passage_probs]

    results = {
        "passages": passages[:k],
        "passage_ids": passage_ids[:k],
        "passage_ranks": passage_ranks[:k],
        "passage_scores": passage_scores[:k],
        "passage_probs": passage_probs[:k],
    }

    return results


@app.route("/search", methods=["GET", "POST"])
def get():
    args = req_parser.parse_args()
    user_query = args["query"]
    evi_num = args["evi_num"]
    try:
        results = search(user_query, evi_num)
    except Exception as e:
        logger.error(str(e))
        return {}, 500

    return results


if __name__ == "__main__":
    app.run(port=5000, debug=False, use_reloader=False)
