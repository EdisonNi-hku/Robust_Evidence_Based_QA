import json
import pandas as pd
import argparse
import random
from transformers import AutoTokenizer

# PROMPT = """ Below is a chat between a curious user and an assistant. The assistant gives helpful, detailed, accurate responses to the user's input.
# ### USER: {instruction}
#
# ### ASSISTANT: """

llama_2 = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

{instruction} [/INST] """

zephyr = """<|system|>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.</s>
<|user|>
{instruction}</s>
<|assistant|>
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default="")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--sample", type=int, default=-1)
    args = parser.parse_args()

    sample_length = len(pd.read_json('train_data/SynSciQA++.json'))
    if 'zephyr' in args.arch:
        template = zephyr
        eos = '</s>'
    else:
        template = llama_2
        eos = ' </s>'
    # tokenizer = AutoTokenizer.from_pretrained(args.arch)

    if args.data_path.endswith('json'):
        with open(args.data_path, 'r') as f:
            data = json.load(f)
        parsed = [{"input": template.format(instruction=d['instruction']), "output": d['response'] + eos} for d in data]
    else:
        df = pd.read_csv(args.data_path)
        instructions = df.instruction.to_list()
        responses = df.response.to_list()
        parsed = [{"input": template.format(instruction=i), "output": r + eos} for i, r in zip(instructions, responses)]

    if args.sample > 0:
        random.seed(42)
        parsed = random.sample(parsed, sample_length)

    print(len(parsed))
    with open(args.output_path, 'w') as f:
        json.dump(parsed, f)