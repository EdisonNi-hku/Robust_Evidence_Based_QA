import argparse
import pandas as pd
import re
import pylcs
import numpy as np
import json
import pickle
from tqdm import tqdm


def split_paragraph(paragraph, parsed_source):
    source_names = list(parsed_source.keys())
    pattern = "|".join([source_n.replace(".", "\.?").replace(",", ",?").replace("(", "\(").replace(")", "\)") for source_n in source_names])

    findall = re.findall(pattern, paragraph)
    split = re.split(pattern, paragraph)
    ref_dict = []
    latest_answer_sent = None
    false_citation = False
    for i in range(len(findall)):
        if bool(re.search(r'\b[A-Za-z]+\b', split[i])):
            ref_dict.append([findall[i], split[i].strip('(').strip(').').strip()])
            latest_answer_sent = split[i].strip('(').strip(').').strip()
        else:
            false_citation = True
            ref_dict.append([findall[i], latest_answer_sent])

    return ref_dict, false_citation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--response_field", type=str, default=None)
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--output_name", type=str, default="")
    parser.add_argument("--spliter", type=str, default='[/INST] ')
    args = parser.parse_args()

    if args.response_field:
        if args.data_path.endswith('json'):
            generated_data = pd.read_json(args.data_path)
        else:
            generated_data = pd.read_csv(args.data_path)
        instructions = generated_data['instruction'].to_list()
        responses = generated_data[args.response_field].to_list()
    else:
        generated_data = pd.read_csv(args.data_path)['output'].to_list()
        instructions = [d.split(args.spliter)[0] for d in generated_data]
        responses = [d.split(args.spliter)[-1] for d in generated_data]
    sources = [d.split('[BEGIN OF SOURCES]')[-1].split('[END OF SOURCES]')[0].strip('\n').split('\n') for d in
               instructions]
    parsed_sources = []
    for source in sources:
        source_dict = {}
        for s in source:
            if ': ' in s:
                source_dict[s.split(":", 1)[0]] = s.split(":", 1)[1]
        parsed_sources.append(source_dict)
    note = []
    parsed_hyps = []
    for s, p in tqdm(zip(responses, parsed_sources), total=len(responses)):
        if len(p) == 0:
            parsed_hyps.append([])
            note.append("No Citation")
        else:
            sent_src_pairs, false_ref = split_paragraph(s, p)
            if not false_ref and len(sent_src_pairs) > 0:
                parsed_hyps.append(sent_src_pairs)
                note.append("Nothing")
            if false_ref:
                parsed_hyps.append([])
                note.append("Illegal Citation")
            if len(sent_src_pairs) == 0:
                parsed_hyps.append([])
                note.append("No Citation")
    if args.response_field:
        filtered_generated_data = pd.DataFrame(
            {'instruction': instructions, 'response': responses, 'note': note, 'index': list(range(len(instructions)))})
    else:
        filtered_generated_data = pd.DataFrame({'response': generated_data, 'note': note, 'index': list(range(len(instructions)))})
    filtered_generated_data.loc[filtered_generated_data['note'] == "Nothing", :].to_csv(args.output_name + '_evident' + '.csv', index=False)
    filtered_generated_data.loc[filtered_generated_data['note'] == "Illegal Citation", :].to_csv(
        args.output_name + '_illegal_citation' + '.csv', index=False)
    filtered_generated_data.loc[filtered_generated_data['note'] == "No Citation", :].to_csv(
        args.output_name + '_no_citation' + '.csv', index=False)

    evidence_answer_pairs = []
    assert (len(parsed_hyps) == len(parsed_sources))
    for hyps, source_dict in tqdm(zip(parsed_hyps, parsed_sources), total=len(parsed_hyps)):
        if len(hyps) == 0:
            continue
        local_evidence_answer = []
        for hyp in hyps:
            source_name = hyp[0]
            answer_sent = hyp[1]
            evidence = source_dict.get(source_name)
            if evidence is None:
                all_source_names = list(source_dict.keys())
                subseq_length = pylcs.lcs_sequence_of_list(source_name, all_source_names)
                longest_common_seq = all_source_names[np.argmax(subseq_length)]
                longest_common_len = max(subseq_length)
                if longest_common_len / len(longest_common_seq) < 0.6:
                    evidence = 'No evidence'
                else:
                    evidence = source_dict.get(longest_common_seq)
            local_evidence_answer.append([evidence, answer_sent])
        evidence_answer_pairs.append(local_evidence_answer)
    parsed_evidence_answer = []
    for i, ls in enumerate(evidence_answer_pairs):
        for l in ls:
            parsed_evidence_answer.append(l + [i])

    df = pd.DataFrame(parsed_evidence_answer, columns=['evidence', 'answer', 'id'])
    df.to_csv(args.output_name + '_paired_answers.csv', index=False, encoding='utf-8')


if __name__ == '__main__':
    main()