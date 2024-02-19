import argparse
import pandas as pd
import torch
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig, set_seed, logging, AutoModelForCausalLM, GenerationConfig

from peft import PeftModel

# PROMPT = """ Below is a chat between a curious user and an assistant. The assistant gives helpful, detailed, accurate responses to the user's input.
# ### USER: {instruction}
#
# ### ASSISTANT: """

LLAMA_TEMPLATE = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

{instruction} [/INST] """

ZEPHYR_TEMPLATE = """<|system|>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.</s>
<|user|>
{instruction}</s>
<|assistant|>
"""


def batchify_list(input_list, batch_size):
    # Calculate the number of batches required
    num_batches = (len(input_list) + batch_size - 1) // batch_size

    # Create empty list to hold batches
    batches = []

    # Generate batches
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(input_list))
        batch = input_list[start_idx:end_idx]
        batches.append(batch)

    return batches


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_cache_dir", type=str, default="cache")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--architecture", type=str, default="llama-2")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-2-*b-hf")
    parser.add_argument("--tokenizer_name", type=str, default="meta-llama/Llama-2-*b-hf")
    parser.add_argument("--prompt_file", type=str, default="example_prompts.json")
    parser.add_argument("--instruction_field", type=str, default="instruction")
    parser.add_argument("--output_file", type=str, default="")
    parser.add_argument("--max_new_token", type=int, default=256)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--load_peft", action="store_true", default=False)
    parser.add_argument("--no_sample", action="store_true", default=False)
    parser.add_argument("--load_8bit", action="store_true", default=False)
    parser.add_argument("--load_tokenizer", action="store_true", default=False)
    return parser.parse_args()


def main(args):
    if args.load_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side='left', local_files_only=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, cache_dir=args.model_cache_dir, padding_side='left', local_files_only=True)
    if args.load_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path if not args.load_peft else args.base_model,
            load_in_8bit=True,
            device_map="auto",
            cache_dir=args.model_cache_dir,
            local_files_only=True,
        )

    else:
        print("Loading 4bit")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path if not args.load_peft else args.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=args.model_cache_dir,
            local_files_only=True,
            load_in_4bit=True,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4',
            )
        )
    if args.load_peft:
        model = PeftModel.from_pretrained(model, args.model_path, local_files_only=True)

    if not args.load_tokenizer:
        if args.architecture == 'llama-1':
            print("Setting EOS, BOS, and UNK tokens for LLama tokenizer")
            tokenizer.add_special_tokens(
                {
                    "eos_token": "</s>",
                    "bos_token": "<s>",
                    "unk_token": "<unk>",
                }
            )
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if args.architecture == 'llama-2':
            print("Add padding token")
            tokenizer.add_special_tokens(
                {
                    "pad_token": "[PAD]",
                }
            )
            model.resize_token_embeddings(len(tokenizer))
    if args.load_peft:
        if args.architecture == 'llama-2':
            model.resize_token_embeddings(len(tokenizer))
    model.eval()

    if args.prompt_file.endswith('csv'):
        df = pd.read_csv(args.prompt_file)
    elif args.prompt_file.endswith('xlsx'):
        df = pd.read_excel(args.prompt_file)
    elif args.prompt_file.endswith('json'):
        df = pd.read_json(args.prompt_file)
    else:
        if args.prompt_file.endswith('jsonl'):
            lines = True
        else:
            lines = False
        df = pd.read_json(args.prompt_file, lines=lines)
    instructions = df[args.instruction_field].to_list()
    if args.architecture == 'llama-2':
        template = LLAMA_TEMPLATE
    else:
        template = ZEPHYR_TEMPLATE
    prompts = [template.format(instruction=p) for p in instructions]
    prompt_batches = batchify_list(prompts, args.batch_size)
    outputs = []
    for batch in tqdm(prompt_batches, total=len(prompt_batches)):
        input_ids = tokenizer(
            batch,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=3072,
        ).to("cuda")
        output_seq = model.generate(
            **input_ids,
            generation_config=GenerationConfig(
                do_sample=not args.no_sample,
                max_new_tokens=args.max_new_token,
                top_p=1 if args.no_sample else args.top_p,
                temperature=1 if args.no_sample else args.temperature,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        )
        output = tokenizer.batch_decode(output_seq, skip_special_tokens=True)
        outputs.extend(output)
    output_df = pd.DataFrame({'output': outputs})
    output_df.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    args = get_args()

    if args.seed >= 0:
        set_seed(args.seed)

    logging.set_verbosity_info()

    main(args)