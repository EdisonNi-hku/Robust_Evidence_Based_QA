import argparse
import pandas as pd
import spacy
import numpy as np

nlp = spacy.load("en_core_web_sm")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--xl_entailment', type=str, default='')
    parser.add_argument('--xxl_entailment', type=str, default='')
    parser.add_argument('--original_data', type=str, default='')
    parser.add_argument('--filtered_data', type=str, default='')
    args = parser.parse_args()

    assert(args.xl_entailment != '' and args.xxl_entailment != '' and args.original_data != '')

    xl_entailment = pd.read_csv(args.xl_entailment)
    xxl_entailment = pd.read_csv(args.xxl_entailment)
    sentence_num = np.array([1 if len(list(nlp(answer).sents)) == 1 else 0 for answer in xl_entailment['answer']])
    filtered = sentence_num * xl_entailment['accuracy'].values * xxl_entailment['accuracy'].values
    filtered_sent = sentence_num
    filtered_xl = sentence_num * xl_entailment['accuracy'].values
    xl_entailment['filtered'] = filtered
    xl_entailment['filtered_sent'] = filtered_sent
    xl_entailment['filtered_xl'] = filtered_xl
    # xl_entailment.loc[:, ['evidence', 'answer', 'id', 'filtered']].to_csv('entailment_test.csv', index=False)

    if args.original_data.endswith('jsonl'):
        original_data = pd.read_json(args.original_data, lines=True)
    elif args.original_data.endswith('json'):
        original_data = pd.read_json(args.original_data, lines=False)
    else:
        original_data = pd.read_csv(args.original_data)

    filtered_indices = set(xl_entailment.loc[xl_entailment['filtered'] == 0, 'id'].to_list())
    remaining_indices = [i for i in range(len(original_data)) if i not in filtered_indices]
    original_data.iloc[remaining_indices].to_csv(args.filtered_data + '_sent_xl_xxl.csv', index=False)


if __name__ == '__main__':
    main()