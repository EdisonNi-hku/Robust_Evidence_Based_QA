### IMPORTANT: We are using openai version 0.28 in these scripts still
# !pip install openai==0.28
import openai
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import random
from time import sleep
import re
import json


# "gpt-3.5-turbo", "gpt-4"
def processPrompt(prompt, model):
    completion = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    answer = completion.choices[0].message.content
    return answer


def createPrompt(question, data, data_all):
    data_sub = data.loc[data["question"] == question]
    data_nonsub = data.loc[data["question"] != question]
    data_all_nonsub = data_all.loc[data_all["question"] != question]

    # sample 0-3 relevant and 3-6 "irrelevant" sources
    rel_int = 0
    if random.randint(0, 4):  # in 20% of the cases, there are actually no right sources given
        rel_int = random.randint(1, len(data_sub))

    irrel_int = random.randint(2, 4)
    irrel_int_all = random.randint(1, 2)
    # sample
    rel_sc = data_sub.sample(n=rel_int)
    irrel_sc = data_nonsub.sample(n=irrel_int)
    irrel_sc_all = data_all_nonsub.sample(n=irrel_int_all)
    srcs = pd.concat([rel_sc, irrel_sc, irrel_sc_all])
    # shuffle the whole thing to not only have one specifc source
    srcs = srcs.sample(frac=1)

    # create a prompt of it in the given style
    pretext = "Given are the following sources: [BEGIN OF SOURCES]"
    posttext = f""" [END OF SOURCES] \n\nCan you respond to the question \"{question}\" by only relying on the sources. Ignore all sources that do not provide an answer to the question. \
                   Do not include any knowledge from outside of these sources. Only write a single paragraph. Each sentence must end with the reference in the form of (author, year, page number). Strictly follow this format. Citing multiple sources in one sentence is not allowed. \
                   However, if no source addresses the question, admit truthfully that no answer can be given. \
                   Answer the question concisely and avoid being verbose."""

    midtext = ""
    for i in range(len(srcs)):
        cite_text = srcs['cite'].iloc[i]
        src_text = srcs['source'].iloc[i]
        midtext += "\n" + cite_text + ": " + src_text

    # create final prompt
    prompt = pretext + midtext + posttext
    return prompt, rel_sc.cite.to_list(), pd.concat([irrel_sc, irrel_sc_all]).cite.to_list(), question


# go through a whole topic and go through the questions
def goThroughTopic(data, data_all):
    dicts = []
    questions = data.question.unique()
    topic = data.topic.unique().tolist()[0]
    # for dataframe
    qs, prs, ans, relscs, irelscs, ms = [], [], [], [], [], []
    for q in questions:
        # print(q)
        # try:
        model = "gpt-3.5-turbo"  # or "gpt-4"
        if random.randint(0, 3):  # 1/4 of the answers are created with gpt4
            model = "gpt-4"
        prompt, rel_sc, irrel_sc, question = createPrompt(q, data, data_all)
        answer = processPrompt(prompt, model)
        temp = {"instruction": prompt, "response": answer, "category": "contextual"}
        # print(temp)
        dicts.append(temp)
        # create dict
        qs.append(question)
        prs.append(prompt)
        ans.append(answer)
        relscs.append(rel_sc)
        irelscs.append(irrel_sc)
        ms.append(model)
        # except:
        # continue
    print(len(dicts))

    # create dataframe and output
    temp_df = pd.DataFrame(
        {"question": qs, "instruction": prs, "response": ans, "right_source": relscs, "other_source": irelscs,
         "model": ms})
    temp_df.to_csv(f'./Data_Final/{topic}.csv')

    # write to JSON in output
    with open(f'./Data_Final/{topic}.json', 'w') as f:
        json.dump(dicts, f)

    return dicts


def main():
    # OpenAI interface call function
    openai.api_key = "sk-"  # YOUR API KEY

    # create all_combined for GPT4 data
    # create df with all instructions and responses
    dfs = []
    names = glob.glob("./Data_Raw/*.csv")
    for n in names:
        temp = pd.read_csv(n, index_col=0)
        dfs.append(temp)

    data_all = pd.concat(dfs)
    # data_all.to_csv("./Data_Final/all_combined.csv")
    # create seperate citation
    data_all["cite"] = data_all.source.apply(
        lambda x: False if not re.search(r'\[(.*?)\]', x) else re.search(r'\[(.*?)\]', x).group(1))
    # delete citation in source
    data_all["source"] = data_all.source.apply(lambda x: re.sub(r'\[(.*?)\]', '', x))
    # delete enumeration if string begins with it
    data_all["question"] = data_all.question.apply(lambda x: re.sub(r'^([1-9]|[1-9][0-9])\.', '', x))

    names = glob.glob("./Data_Raw/*")
    for name in names:
        if "all_combined.csv" in name:
            continue
        print(name)
        df = pd.read_csv(name, index_col=0)
        # create seperate citation
        df["cite"] = df.source.apply(
            lambda x: False if not re.search(r'\[(.*?)\]', x) else re.search(r'\[(.*?)\]', x).group(1))
        # delete citation in source
        df["source"] = df.source.apply(lambda x: re.sub(r'\[(.*?)\]', '', x))
        # delete enumeration if string begins with it
        df["question"] = df.question.apply(lambda x: re.sub(r'^([1-9]|[1-9][0-9])\.', '', x))
        goThroughTopic(df, data_all)

if __name__ == '__main__':
    main()
