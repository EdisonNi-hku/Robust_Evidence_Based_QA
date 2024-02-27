### IMPORTANT: We are using openai version 0.28 in these scripts still
# !pip install openai==0.28
import openai
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import random
from time import sleep
import time
import re

# create random questions for a topic
def createQuestions(topic, n, model):
  q_prompt = f""" Take the topic {topic} and create {n} questions that could be posed in the field. Make the questions diverse and differentiable from each other.

                  End every question with '\\'. Use no enumeration or additional signs to seperate the questions.
              """
  answer = processPrompt(q_prompt, model)
  answer = answer.replace("\n", "")
  questions = answer.split("\\")
  questions = [x for x in questions if x != ""]
  questions = [re.sub(r'^([1-9]|[1-9][0-9])\.', '', x).strip() for x in questions]
  return questions

# create sources to a given topic
def createSources(topic, m, question, model):
  s_prompt = f"""
              Consider the following question within the topic {topic}: {question}

              Please create {m} paragraphs with the length of 2-4 sentences that partially address this question. The question should not fully be answered by one paragraph but rather helpful content in respect to the question should be displayed. \
              Each paragraph should be in the style of a book or research article. \
              Furthermore, the paragraphs can display different perspectives and should not overlap much. The paragaphs should also alternate in level of detail and addressed readers, \
              i.e., some paragraphs can be very scientifc while others would rather serve a general public. \
              It is important that the paragraphs stand for themselves. They don't read like one article but excerpts from multiple articles. \
              Please be creative with the beginning of the paragraphs.

              In the end of each paragraph give author, year and page in the following format '[author, year, page]'. Follow this example: '[Mishra et al., 2019, p.54]'. \
              Make up author, year and page, if you don't have this information. Authors can also be institutions.

              End every paragaph with 'ENDOFPARAGRAPH'. Use no enumeration or additional signs to seperate the paragraphs. Also do not give any further information like "Paragraph 1: ...".
              """
              # Please be creative with the author names, don't just use common English names.

  answer = processPrompt(s_prompt, model)
  answer = answer.replace("\n", "")
  sources = answer.split("ENDOFPARAGRAPH") # seperate the individual answers
  sources = [x for x in sources if x != ""] # exclude empty answers
  sources = [re.sub(r'^(Paragraph [0-9]:)', '', x).strip() for x in sources]
  return sources

# create dataset for each topic and question
def createTopicData(topic, n, m, model):
  # save questions and sources
  ts = []
  qs = []
  ss = []
  # get questions
  questions = createQuestions(topic, n, model)
  # create sources for each question
  for q in questions:
    # if it doesnt work, retry in 1 seconds
    print(q)
    try:
      sources = createSources(topic, m, q, model)
    except:
      done = False
      while not done:
          print("Failed once. Wait")
          sleep(20)
          try:
              sources = createSources(topic, m, q, model)
              done = True
          except:
              print("Failed again. Wait again.")
              sleep(20)
              continue
    # if sources are there then store
    for s in sources:
      ts.append(topic)
      qs.append(q)
      ss.append(s)

  # create dataset that store everything
  df = pd.DataFrame({"topic": ts, "question": qs, "source": ss})
  ### SAVE AT A PREFINED DESTINATION
  df.to_csv(f"./Data_Raw/{topic}_question_source.csv")
  return df

def main():
    # OpenAI interface call function
    openai.api_key = "sk-" # YOUR API KEY
    model = "gpt-3.5-turbo"

    # create random topics
    n = 110
    prompt = f"Create {n} random topics from the scientific areas of finance, sustainability, physics, social sciences and natural sciences. Please seperate each topic with '||'. Use no enumeration or additional signs to seperate the topics."
    answer = processPrompt(prompt, model)
    topic_list = answer.split("\n")
    topic_list = [re.sub(r'^([1-9]|[1-9][0-9])\.', '', x).strip() for x in topic_list]

    # define number of questions and paragraphs per question
    num_questions = 25
    num_paragraphs = 3

    for top in topic_list:
        df = createTopicData(top, num_questions, num_paragraphs, model)

if __name__ == '__main__':
    main()