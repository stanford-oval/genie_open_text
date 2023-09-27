"""Test the performance of the ColBERT retrieval part."""
import sys
from argparse import ArgumentParser

sys.path.insert(0, "./ColBERT/")

from colbert.data import Queries, Collection
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--collection_path', type=str, help='Path of the collection in tsv format.')
    parser.add_argument('--index_path', type=str, help='Path of the colbert index.')
    parser.add_argument('--model_dir', type=str, help='Path of the colbert model.')
    parser.add_argument('--output_file', type=str, help='File to log the query and retrieval results.')
    args = parser.parse_args()

    def log_output(s):
        with open(args.output_file, mode='a') as f:
            f.write(s+'\n')

    with Run().context(RunConfig(experiment="cooking", index_root="")):

        collection = Collection(path=args.collection_path)
        searcher = Searcher(index=args.index_path, checkpoint=args.model_dir, collection=collection)
        while True:
            query = input("Query: ")
            log_output(query)
            results = searcher.search(query, k=3)
            for p_id, p_rank, p_score in zip(*results):
                line = f"\t [{p_rank}] \t\t {p_score:.1f} \t\t {searcher.collection[p_id]}"
                print(line)
                log_output(line)

            log_output('\n#################')
