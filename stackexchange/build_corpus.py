"""
Process each question-answer pair with time stamp into a single document.

collection format for ColBERT: each line is `pid \t passage text` (saved as .tsv file)
"""
import os
import pickle
from argparse import ArgumentParser

import pandas as pd
from tqdm import tqdm


def main(args):
    path = args.input

    with open(os.path.join(path, 'Answers.pkl'), 'rb') as file:
        answers = pickle.load(file)  # {ans_id: ans_text}
    with open(os.path.join(path, 'Questions.pkl'), 'rb') as file:
        questions = pickle.load(file)  # {q_id: [q_text, q_description]}
    if args.add_comment:
        with open(os.path.join(path, 'PostId_CommenterId_Text.pkl'), 'rb') as file:
            comments = pickle.load(file)
            # {"PostId": [ans/q_id], "UserId": [], "Score": [], "Text": [], "CreationDate": []}
    ans2q = pd.read_csv(os.path.join(path, 'AnswerId_QuestionId.csv'))
    # schema: QuestionId, AnswerId, Score, CreationDate

    docs = []
    err_cnt = 0
    neg_voted_ans_cnt = 0

    for _, row in tqdm(ans2q.iterrows()):
        q_id = row['QuestionId']
        ans_id = row['AnswerId']
        if q_id not in questions or ans_id not in answers:
            err_cnt += 1
            continue
        if row['Score'] < 0:
            neg_voted_ans_cnt += 1
            continue  # We delete answers which have negative voting for they are viewed as untrusted.
        q = questions[q_id]
        ans = answers[ans_id]
        if args.add_comment:
            raise NotImplementedError  # TODO: Save for later.
        else:
            doc = f"Question: {q[0]} Context of the question: {q[1]} | " \
                  f"Answer at {row['CreationDate']} [voting={row['Score']}]: {ans}"
            doc = doc.replace('\n', ' ')
            doc = doc.replace('\r', ' ')
            doc = doc.replace('\t', ' ')
        docs.append(doc)

    print(f'#Documents = {len(docs)}')
    print(f'#Pair with ERROR = {err_cnt}')
    print(f'#Negatively voted answers = {neg_voted_ans_cnt}')
    df = pd.DataFrame({'pid': range(len(docs)), 'doc': docs})
    df.to_csv(os.path.join(path, f'{args.domain}_collection.tsv'), sep='\t', index=False, header=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, help='Input data path.')
    parser.add_argument('--domain', type=str, help='Domain name.')
    parser.add_argument('--add_comment', action='store_true',
                        help='Whether to include comments under a question/answer.')
    parser.add_argument('--mode', choices=['colbert'], default='colbert')

    main(parser.parse_args())
