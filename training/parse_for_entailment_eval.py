import spacy
import argparse
import pandas as pd
import re
from tqdm import tqdm
import numpy as np
nlp = spacy.load('en_core_web_sm')


CITATION_REGEX = r'\b[A-Za-z].+?,\s*\d{4},\s*p\.\s*\d+'


def ends_with_parenthesis(sentence):
    # Strip trailing spaces
    sentence = sentence.strip()

    # Check if the last character is a parenthesis
    if sentence.endswith(')'):
        return True

    # Check for a closing parenthesis before the last punctuation mark
    for i in range(len(sentence) - 1, -1, -1):
        if sentence[i] in '.,;!?':
            continue
        return sentence[i] == ')'

    return False


def extract_outer_parenthesis_content(sentence):
    # Strip trailing spaces and any trailing punctuation
    sentence = sentence.rstrip(" .,;!?")
    num_close = 0
    end_idx = len(sentence)
    for i in range(len(sentence) - 1, -1, -1):
        if sentence[i] == ')':
            num_close += 1
            if num_close == 1:
                end_idx = i
        elif sentence[i] == '(' and num_close == 1:
            return sentence[i+1:end_idx]
        if sentence[i] == '(' and num_close > 1:
            num_close -= 1
    return None


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
        generated_data = pd.read_csv(args.data_path)
        instructions = [d.split(args.spliter)[0] for d in generated_data['output']]
        responses = [d.split(args.spliter)[-1] for d in generated_data['output']]
    sources = [d.split('[BEGIN OF SOURCES]')[-1].split('[END OF SOURCES]')[0].strip('\n').split('\n') for d in
               instructions]
    parsed_sources = []
    for source in sources:
        source_dict = {}
        for s in source:
            if ': ' in s:
                source_dict[s.split(":", 1)[0]] = s.split(":", 1)[1]
        parsed_sources.append(source_dict)

    response_sentences = []
    response_id = []
    format_correctness = []
    format_error_type = []
    no_citation_id = []
    source_texts = []
    for i, r in tqdm(enumerate(responses)):
        citations = re.findall(CITATION_REGEX, r)
        if len(citations) == 0:
            no_citation_id.append(i)
            continue
        doc = nlp(r)
        sent_list = [s.text for s in doc.sents]
        for sent in sent_list:
            response_sentences.append(sent)
            response_id.append(i)
            parenthesis = ends_with_parenthesis(sent)
            if parenthesis:
                parenthesis_content = extract_outer_parenthesis_content(sent)
                if parenthesis_content is None:
                    format_correctness.append(0)
                    format_error_type.append(1)
                    source_texts.append('format error')
                    continue
                all_citations = re.findall(CITATION_REGEX, parenthesis_content)
                if len(all_citations) == 1:
                    source_text = parsed_sources[i].get(all_citations[0])
                    if source_text is not None:
                        format_correctness.append(1)
                        format_error_type.append(0)
                        source_texts.append(source_text)
                    else:
                        if 'p. ' in all_citations[0]:
                            source_text = parsed_sources[i].get(all_citations[0].replace('p. ', 'p.'))
                        else:
                            source_text = parsed_sources[i].get(all_citations[0].replace('p.', 'p. '))
                        if source_text is not None:
                            format_correctness.append(1)
                            format_error_type.append(0)
                            source_texts.append(source_text)
                        else:
                            format_correctness.append(0)
                            format_error_type.append(2)
                            source_texts.append(all_citations[0] + ' fails to match source name.')
                else:
                    format_correctness.append(0)
                    format_error_type.append(3)
                    source_texts.append('multiple citations')
            else:
                format_correctness.append(0)
                format_error_type.append(1)
                source_texts.append('format error')
    generated_data.iloc[no_citation_id].to_csv(args.output_name + '_no_citation' + '.csv', index_label='id')
    with_citation = pd.DataFrame({'answer': response_sentences, 'response_id': response_id, 'format': format_correctness, 'format_error_type': format_error_type, 'evidence': source_texts})
    with_citation.to_csv(args.output_name + '_with_citation' + '.csv', index_label='id')
    with_citation.loc[with_citation['format'] == 1, :].to_csv(args.output_name + '_paired_answers' + '.csv', index_label='id')


if __name__ == '__main__':
    main()

