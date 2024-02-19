import argparse
import pandas as pd
import transformers
from typing import Dict
from tqdm import tqdm
import torch
from inference import batchify_list
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig, AutoModelForCausalLM, BitsAndBytesConfig

PROMPT = "As an Attribution Validator, your task is to verify whether a given reference can support the given claim. A claim can be either a plain sentence or a question followed by its answer. Specifically, your response should clearly indicate the relationship: Attributable, Contradictory or Extrapolatory. A contradictory error occurs when you can infer that the answer contradicts the fact presented in the context, while an extrapolatory error means that you cannot infer the correctness of the answer based on the information provided in the context. \n\nClaim: {claim_sentence} \n Reference: {evidence_sentence}"
PROMPT_vi = 'Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:'.format("Verify whether a given reference can support the claim. Options: Attributable, Extrapolatory or Contradictory. Attributable means the reference fully supports the claim, Extrapolatory means the reference lacks sufficient information to validate the claim, and Contradictory means the claim contradicts the information presented in the reference.", "Claim: {claim_sentence}\n\nReference: {evidence_sentence}\n")


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_cache_dir", type=str, default="/home/jini/shares/transformer_models")
    parser.add_argument("--input_file", type=str, default="*.csv")
    parser.add_argument("--output_file", type=str, default="*.csv")
    parser.add_argument("--model_name", type=str, default="osunlp/attrscore-flan-t5-xl")
    parser.add_argument("--load_4bit", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.model_cache_dir)
    if 't5' in args.model_name:
        model_cls = AutoModelForSeq2SeqLM
        prompt_template = PROMPT
        generation_config = GenerationConfig(
            max_new_tokens=5,
        )
    else:
        model_cls = AutoModelForCausalLM
        prompt_template = PROMPT_vi
        generation_config = GenerationConfig(
            temperature=0,
            top_p=0.9,
            num_beams=1,
            do_sample=False,
            max_new_tokens=64,
        )
    if not args.load_4bit:
        model = model_cls.from_pretrained(args.model_name, cache_dir=args.model_cache_dir).to("cuda")
    else:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        model = model_cls.from_pretrained(args.model_name, quantization_config=quantization_config, cache_dir=args.model_cache_dir)
    if 't5' not in args.model_name:
        DEFAULT_PAD_TOKEN = "[PAD]"
        DEFAULT_EOS_TOKEN = "</s>"
        DEFAULT_BOS_TOKEN = "<s>"
        DEFAULT_UNK_TOKEN = "<unk>"
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                tokenizer=tokenizer,
                model=model,
            )
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )
    model.eval()
    input_df = pd.read_csv(args.input_file)
    pairs = zip(input_df['evidence'].to_list(), input_df['answer'].to_list())
    prompts = [prompt_template.format(claim_sentence=p[1], evidence_sentence=p[0]) for p in pairs]
    prompt_batches = batchify_list(prompts, args.batch_size)
    outputs = []
    for batch in tqdm(prompt_batches, total=len(prompt_batches)):
        input_ids = tokenizer(
            batch,
            return_tensors='pt',
            padding=True,
            truncation=True,
        ).to("cuda")
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            output_seq = model.generate(
                **input_ids,
                generation_config=generation_config,
            )
        output = tokenizer.batch_decode(output_seq, skip_special_tokens=True)
        outputs.extend(output)
    input_df['attribution'] = outputs
    accuracy = []
    if 't5' in args.model_name:
        input_df['accuracy'] = input_df['attribution'].apply(lambda x: 1 if "Attributable" in x.strip() else 0)
    else:
        input_df['accuracy'] = input_df['attribution'].apply(lambda x: 1 if "Attributable" in x.split("### Response:")[-1] else 0)
    print("Accuracy:", input_df['accuracy'].mean())

    input_df.to_csv(args.output_file, index=False)