import pandas as pd
import re
import json
import argparse

# function that takes the raw input and identifies the right sources (those that are certainly connected to the answer)
def getRightSources(instruction, scientific_data):
  # search for question in instruction (maybe instruction and response might be combined)
  question = re.search(r'Can you respond to the question "(.*?)" by only relying on the sources', instruction).group(1)
  # search for question in data
  sub = scientific_data.loc[scientific_data.question.apply(lambda x: question in x)]

  # find source names in instruction
  source_long = []
  for source in sub.source:
    source = re.split('\+|\,|\;|\(|\)', source[:30])[0]
    try:
      match_here = re.search(fr'\n(.*?):\s*{source}', instruction)
    except:
      print("Error")
      match_here = 0

    if match_here:
      source_long.append(match_here.group(1))

  # find other sources
  all_sources = re.findall(r"\n(.*p\.\s?\d{1,5}):", instruction)
  other_sources = []
  for i in all_sources:
    if i in source_long:
      continue
    other_sources.append(i)

  return question, source_long, other_sources

# function that finds the occurance of the right sources in the response
# exact match, author and year, only authors
def findSourcesInResponse(response, sources, other_sources):
  out = {}
  os_out = {}
  for s in sources:
    s_sub = {}
    author = s.split(",")[0]
    year = s.split(",")[1].replace(" ", "")
    author_year = s.split(",")[0] + "," + s.split(",")[1]
    s_sub["author_appearance"] = "yes" if author in response else "no"
    s_sub["author_year_appearance_post"] = "yes" if author_year in response else "no"
    s_sub["author_year_appearance_intext"] = "yes" if re.search(fr'{author}\s?\({year}', response, re.IGNORECASE) else "no"
    s_sub["all_appearance"] = "yes" if ((s in response) or (s.replace("p. ", "p.") in response) or (s.replace("p.", "p. ") in response)) else "no"
    out[s] = s_sub

  for os in other_sources:
    os_sub = {}
    author = os.split(",")[0]
    year = os.split(",")[1].replace(" ", "")
    author_year = os.split(",")[0] + "," + os.split(",")[1]
    os_sub["author_appearance"] = "yes" if author in response else "no"
    os_sub["author_year_appearance_post"] = "yes" if author_year in response else "no"
    os_sub["author_year_appearance_intext"] = "yes" if re.search(fr'{author}\s?\({year}', response, re.IGNORECASE) else "no"
    os_sub["all_appearance"] = "yes" if ((os in response) or (os.replace("p. ", "p.") in response) or (os.replace("p.", "p. ") in response)) else "no"
    os_out[os] = os_sub

  return out, os_out

# this function evaluates the outputed source dictionaries
# we take the source dict and give which elements are counting as rightfully cited (e.g. only 'all_appearance' => ['all_appearance'])
def evaluate_sources(source_dict, list_of_counting_elements):
  count = 0
  for k in source_dict.keys():
    add = False
    for ce in list_of_counting_elements:
      if source_dict[k][ce] == "yes":
        add = True
    if add:
      count += 1
  return -1 if len(source_dict.keys()) == 0 else count / len(source_dict.keys())

# evaluate all
def evaluateSourceScientific(name_set, raw_question_data, list_of_counting_elements_rightsource, list_of_counting_elements_othersource):
  # testing texts
  # could be a json with instruction-answer pairs or a csv
  with open(name_set, 'r') as f:
    texts = json.load(f)

  # safe results
  index_in_testset, questions, responses, right_sources, other_sources, cited_right, cited_other, right_sources_count, other_sources_count, instructions = [], [], [], [], [], [], [], [], [], []

  for i, t in enumerate(texts):
    question, source_long, source_other = getRightSources(t["instruction"], raw_question_data)
    out, os_out = findSourcesInResponse(t["response"], source_long, source_other)
    # build dataframe
    index_in_testset.append(i)
    questions.append(question)
    responses.append(t["response"])
    right_sources.append(source_long)
    other_sources.append(source_other)
    cited_right.append(evaluate_sources(out, list_of_counting_elements_rightsource))
    cited_other.append(evaluate_sources(os_out, list_of_counting_elements_othersource))
    right_sources_count.append(len(source_long))
    other_sources_count.append(len(source_other))
    instructions.append(t["instruction"])

  temp = pd.DataFrame({"ind": index_in_testset, "question": questions, "response": responses, "right_source": right_sources, "other_source": other_sources, "cited_right": cited_right,
                       "cited_other": cited_other, "right_sources_count": right_sources_count, "other_sources_count": other_sources_count, "instruction": instructions})
  return temp


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--question_source_pairs", type=str,
                      default="./filter_with_sourceQuality/SynSciQA_all_raw_question_source_pairs.csv")
  parser.add_argument("--name_to_filter", type=str,
                      default='../train_data/SynSciQA.json')
  parser.add_argument("--output_name", type=str,
                      default='./filter_with_sourceQuality/ScientificQA_filtered_test.json')
  args = parser.parse_args()

  # load all raw question source pairs
  # using these leaves flexibility on how you create the dataset
  # you could vary creating SynSciQA by for instance sampling more relevant sources
  question_source_pairs = pd.read_csv(args.question_source_pairs)

  # create a filtered version of SynSciQA where the source quality is ensured
  name_to_filter = args.name_to_filter

  # filtering options:
  # author_appearance: author appear in the answer
  # author_year_appearance_post: author and year appear after the sentence, e.g. "(Schimanski et al., 2024)"
  # author_year_appearance_intext: author and year appear in-text, e.g. "According to Schimanski et al. (2024)"
  # all_appearance: author, year and page appear accordingly to the instruction format, e.g. "Schimanski et al., 2024, p.4"
  # These options enable to filter differently according to your needs.
  option_right_source = ['all_appearance']
  option_other_source = ['all_appearance',
                         'author_year_appearance_intext']

  scientific_GPT_eval_source = evaluateSourceScientific(name_to_filter, question_source_pairs, option_right_source, option_other_source)

  # take out samples where model rejects to answer but there is at least one relevant evidence
  # Explore additional filtering options.
  scientific_GPT_eval_source_filter = scientific_GPT_eval_source[~(
            (scientific_GPT_eval_source["cited_other"] == 0) & (scientific_GPT_eval_source["cited_right"] == 0) & (
              scientific_GPT_eval_source["right_sources_count"] > 0))]
  # take out samples where other sources were cited
  scientific_GPT_eval_source_filter2 = scientific_GPT_eval_source_filter[
    scientific_GPT_eval_source_filter["cited_other"] == 0]

  print("Dataset columns for using different filtering steps.")
  print(scientific_GPT_eval_source.columns)

  # create this quality dataset
  out_clean_sourcing = []
  for index, row in scientific_GPT_eval_source_filter2.iterrows():
    temp_dict = {}
    temp_dict["instruction"] = row["instruction"]
    temp_dict["response"] = row["response"]
    out_clean_sourcing.append(temp_dict)

  with open(f'{args.output_name}', 'w') as f:
    json.dump(out_clean_sourcing, f)
    print(f"Saved data as {args.output_name}")


if __name__ == '__main__':
    main()