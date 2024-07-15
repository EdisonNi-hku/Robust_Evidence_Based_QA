import argparse
import pandas as pd
import numpy as np
import re

"""
An improvement of this method would be to use unique identifiers instead of source names.
For instance, one could transform "Ho et al., 2020, p.76" to "[[SOURCE1]]". This would ease
further processing.
"""

"""
Function that prepares the answers of LLama2 models. Since Llama2 has a certain format, this function might
look different for different models.
"""
def prepare_llama2(dataset_name, golden_name):
  # create dataset
  golden = pd.read_csv(golden_name)
  golden["all_srcs"] = golden.apply(lambda row: re.findall(r"[\"']([\w\d\s]+, \d{4}, p. ?\d*)[\"']", row.right_source) + re.findall(
      r"[\"']([\w\d\s]+, \d{4}, p. ?\d*)[\"']", row.other_source), axis=1)

  ### LLama2-seppfic part
  # might need to be altered for other models
  df_temp = pd.read_csv(dataset_name)
  df_temp["instruction"] = df_temp.output.apply(lambda x: x.split("[/INST]")[0].split("<</SYS>>")[1])
  df_temp["response"] = df_temp.output.apply(lambda x: x.split("[/INST]")[1])
  df_temp["question"] = df_temp["output"].apply(lambda x: re.search(r'Can you respond to the question "(.*?)" by only relying on the sources', x).group(1))
  df_temp = df_temp[["instruction", "question", "response"]]

  # combine with golden
  golden = golden[["question", "right_source", "other_source", "all_srcs"]].copy()
  out = df_temp.set_index("question").join(golden.set_index("question")).reset_index()

  return out

def source_quality(df, response="response"):
  df["regex_allsrcs"] = df["all_srcs"].apply(lambda x: '|'.join(x))

  #or (os.replace("p. ", "p.") in response) or (os.replace("p.", "p. ") in response))
  def findSources(regex, response):
    # vary the regex
    l1 = np.unique(re.findall(rf"{regex}", response)).tolist()
    r2 = regex.replace("p. ", "p.")
    l2 = np.unique(re.findall(rf"{r2}", response)).tolist()
    r3 = regex.replace("p.", "p. ")
    l3 = np.unique(re.findall(rf"{r3}", response)).tolist()
    # output
    lout = np.unique(l1 + l2 + l3)
    return lout

  df["reponse_source"] = df.apply(lambda row: findSources(row['regex_allsrcs'], row[response]), axis=1)
  df["right_sources_count"] = df["right_source"].apply(lambda x: len(re.findall("\'", x)) / 2)

  def helper(cited_sources, right_sources):
    for cs in cited_sources:
      if not cs in right_sources:
        return 1
    return 0

  df["cited_other"] = df.apply(lambda row: helper(row["reponse_source"], row["right_source"]), axis=1)

  others, nosources, somesources, models = [], [], [], []
  eval = df.copy()

  # first measure: not citing others
  # Main source quality score
  others.append(eval[eval["cited_other"] == 0].shape[0] / eval.shape[0])
  # third meassure: detecting the no sources, no answer
  no_source = eval[eval["right_sources_count"] == 0].copy()
  nosources.append(no_source[no_source["cited_other"] == 0].shape[0] / no_source.shape[0])
  # forth meassure: detecting the some sources, right answer
  some_source = eval[eval["right_sources_count"] != 0].copy()
  somesources.append(some_source[some_source["cited_other"] == 0].shape[0] / some_source.shape[0])

  # data output
  data = pd.DataFrame({"source_quality": others, "source_quality_nosource": nosources, "source_quality_somesource": somesources})
  return data

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--golden_data", type=str,
                      #default="./golden_sources/SynSciQA_test_with_golden_sources.csv")
                      default="./golden_sources/GenSearch_test_with_golden_sources.csv")
  parser.add_argument("--model_data", type=str,
                      #default="./sourceQuality_score_test/GenSearch_test_Llama2_answers.csv")
                      default="./sourceQuality_score_test/GenSearch_test_Llama2_answers.csv")
  args = parser.parse_args()

  # combine the golden_data with the model_data
  df = prepare_llama2(args.model_data, args.golden_data)
  score = source_quality(df)
  print(score)

if __name__ == '__main__':
    main()