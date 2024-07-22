# Robust Evidence-Based QA
This repository contains the code for the paper "[Towards Faithful and Robust LLM Specialists for Evidence-Based Question-Answering](https://arxiv.org/pdf/2402.08277.pdf)", where we propose a data synthesize pipeline for fine-tuning and evaluating LLMs for Evidence-Based QA. Fine-tuning on our synthetic data drastically improve the answer traceability and attributability of LLM outputs.

## Data Generation Pipeline
This section introduces how to generate diversified Evidence-Based QA data for training and evaluation. 

The data generation follows five general steps:
1. Generate a broad array of 100+ scientific topics. 
2. Generate 25 distinctive questions for each topic. 
3. Create three source paragraphs _relevant_ to each question. 
4. Design an instruction encompassing 0-3 _relevant_ sources and 3-6 _irrelevant_ sources, along with the corresponding question. 
5. Create an answer to the question following the provided instruction.

For details on the data generation pipeline, see the folder "data_generation". You will need to insert your own Open-AI API key to execute the code.

## Data quality filtering
Following commands show how to apply data quality filters to improve synthetic data quality, leading to superior fine-tuning outcome. In other words, how to obtain **train_data/SynSciQA+.json** and **train_data/SynSciQA++.json** from **train_data/SynSciQA.json**
```shell
## Filter SynSciQA with source quality filter, obtaining SynSciQA+
# These steps show the exemplary filtering process. 
# It is highly recommended to get into the code and understand the options.

# Filtering SynSciQA with the source quality filter (example)
python data_filtering/filter_with_sourceQuality.py --name_to_filter ./train_data/SynSciQA.json --question_source_pairs ./data_filtering/filter_with_sourceQuality/SynSciQA_all_raw_question_source_pairs.csv --output_name ./data_filtering/filter_with_sourceQuality/ScientificQA_filtered_test.json

# Evaluate fine-tuned model output with source quality score (example)
python data_filtering/evaluate_sourceQuality.py --golden_data ./data_filtering/golden_sources/GenSearch_test_with_golden_sources.csv --model_data ./data_filtering/sourceQuality_score_test/GenSearch_test_Llama2_answers.csv

## Filter SynSciQA+ with format and attributability quality, obtaining SynSciQA++

# This script divides original SynSciQA+ into three parts: SynSciQA+_no_citation.csv (the teacher model refuse to answer due to no relevant source); 
# SynSciQA+_illegal_citation.csv (answers with format error, such as no citation or format-wrong citation for some sentences); and
# SynSciQA+_paired_answers.csv (answers with format-correct citations, but need attributability check)
python data_filtering/parse_for_entailment.py --response_field 'response' --data_path 'train_data/SynSciQA+.json' --output_name SynSciQA+

# Evaluate the attributability (entailment) using osunlp/attrscore-flan-t5-xl
python data_filtering/evaluate_entailment.py --input_file SynSciQA+_paired_answers.csv --model_name osunlp/attrscore-flan-t5-xl --ouptut_file SynSciQA+_paired_answers_3b_entail.csv

# Evaluate the attributability (entailment) using osunlp/attrscore-flan-t5-xxl
python data_filtering/evaluate_entailment.py --input_file SynSciQA+_paired_answers.csv --model_name osunlp/attrscore-flan-t5-xxl --ouptut_file SynSciQA+_paired_answers_11b_entail.csv

# Collect data that predicted attributable by both Flan-t5 models
python data_filtering/filter_with_entailment.py --xl_entailment SynSciQA+_paired_answers_3b_entail.csv --xxl_entailment SynSciQA+_paired_answers_11b_entail.csv --original_data data/SynSciQA+.json --filtered_data SynSciQA++
# This will resulting in SynSciQA++_xl_xxl.csv which contains all attributable (predicted by flan-t5 models) instruction-answer pairs.
# Finally, don't forget to merge SynSciQA++_xl_xxl.csv with SynSciQA+_no_citation.csv to obtain SynSciQA++.json :)
```

## Training and Evaluation
Following commands show the steps to reproduce fine-tuning and evaluation results in Table 1, Figure 4, 5, 6, and 7 of the paper.
```shell
# We use the original qlora.py fine-tuning script from https://github.com/jondurbin/qlora.
# We first parse the fine-tuning data into an input-output form depending on the model instruction template.
python training/parse_to_input_output.py --arch zephyr --data_path data/SynSciQA.json --output_path SynSciQA_zephyr.json --sample -1
# --arch can be either zephyr or llama2
# --sample can be either -1 or 1. 
# If sample == 1, the output file will be a down-sampled version has the same size as SynSciQA++ (i.e., SynSciQA+_S and SynSciQA_S in the paper).

# Then run qlora fine-tuning, remember to set the model_name, cache_dir, data_dir, and output_dir.
# The following command results in five epochs of zephyr-7b-beta fine-tuned on SynSciQA, checkpoints stored in zephyr_synsciqa.
bash run_qlora.sh "HuggingFaceH4/zephyr-7b-beta" "cache" "SynSciQA_zephyr.json" "zephyr_synsciqa"

# Let's now evaluate the resulting checkpoints' attributability score on different test sets.
# First inference the answers using fine-tuned checkpoints, for example:
python training/inference.py --architecture zephyr --no_sample --model_cache_dir /path/to/cache --max_new_token 2048 \
 --temperature 0.7 --top_p 0.9 --batch_size 4 --seed 42 --load_tokenizer --base_model HuggingFaceH4/zephyr-7b-beta \
 --load_peft --model_path zephyr_synsciqa/checkpoint-67 --prompt_file test_data/SynSciQA_test.csv --output_file zephyr_synsciqa_synsciqa.csv
# Here we inference SynSciQA_test with the first-epoch checkpoint of zephyr fine-tuned on SynSciQA. 
# The output file is zephyr_synsciqa_synsciqa.csv

# Then compute its source quality score (example)
python data_filtering/evaluate_sourceQuality.py --golden_data ./data_filtering/golden_sources/GenSearch_test_with_golden_sources.csv --model_data ./data_filtering/sourceQuality_score_test/GenSearch_test_Llama2_answers.csv
# Finally compute its attributability score (when the used model is llama-2, spliter should be '[/INST] ' instead)
OUTPUT_NAME=zephyr_synsciqa_synsciqa
python training/parse_for_entailment_eval.py --data_path zephyr_synsciqa_synsciqa.csv --output_name ${OUTPUT_NAME} --spliter="<|assistant|>"
# Predict attributability with Flan-t5 models
python data_filtering/evaluate_entailment.py --load_4bit --input_file ${OUTPUT_NAME}_paired_answers.csv --output_file ${OUTPUT_NAME}_11b_entail.csv --batch_size 4 --model_name osunlp/attrscore-flan-t5-xxl
python data_filtering/evaluate_entailment.py --load_4bit --input_file ${OUTPUT_NAME}_paired_answers.csv --output_file ${OUTPUT_NAME}_3b_entail.csv --batch_size 4 --model_name osunlp/attrscore-flan-t5-xl
# This final command will print out entailment scores.
python training/compute_entailment_scores.py --filename ${OUTPUT_NAME}
```
## Data
We disclose all LLM generations, human annotations, and raw data in paper, which can be found in the following directories:

```markdown
train_data/               # Training Datasets (see Section 3)
│
├── SynSciQA.json         # raw SynSciQA training data
├── SynSciQA+.json        # + applied Source Quality filter
└── SynSciQA++.json       # + applied Answer Attributability Quality Filter

test_data/                # Evaluations Datasets (see Section 4)
│
├── SynSciQA_test.csv     # SynSciQA test set
├── GenSearch_test.csv    # GenSearch test set
├── ChatReport_test.csv   # ChatReport test set
└── ClimateQA_test.csv    # Climate QA test set

data_filtering/                                 # Data for Source Quality
│
├── filter_with_sourceQuality/                  # All generated sources for a question
│   ├── ...
│   └── SynSciQA_all_raw_question_source_pairs.csv
├── golden_sources/                             # Evaluation data for Source Quality
│   ├── GenSearch_test_with_golden_sources.csv  # GenSearch_test with golden sources
│   └── SynSciQA_test_with_golden_sources.csv   # SynSciQA with golden sources
└── ...

handeval_data/                                  # Hand-evaluation data
│
├── handeval_SynSciQA++/                        # Hand-evaluation of quality-filtered data (see Appendix D)
│   └── HandEval_of_QualityFilters_SynSciQA++.xlsx
├── handeval_models_on_benchmark_datasets/      # Hand-evaluation of different models output vs. automatic scores (see Appendix J)
│   ├── ChatReport_test_eval.xlsx               # on ChatReport test set
│   ├── ClimateQA_test_eval.xlsx                # on ClimateQA test set
│   ├── GenSearch_test_eval.xlsx                # on GenSearch test set
│   └── SynSciQA_test_eval.xlsx                 # on SynSciQA test set
```

### Fields of data files

- SynSciQA(+/++).json
  - instruction: instruction given to the model
  - response: response produced by the model

- SynSciQA_test/GenSearch_test/ChatReport_test/ClimateQA_test.csv
  - instruction: instruction given to the model
  - gpt-35: response by GPT-3.5
  - gpt-4: response by GPT-4

- SynSciQA_all_raw_question_source_pairs.csv
  - topic: topic of the question
  - question: question generated
  - source: source generated

- GenSearch_test_with_golden_sources/SynSciQA_test_with_golden_sources.csv
  - question: question
  - right_source: sources that address the question
  - other_source: source that do not address the question
  - instruction: final instruction including question, right_source and other_source


- HandEval_of_QualityFilters_SynSciQA++.xlsx
  - evidence: source paragraph that is cited
  - answer: answer sentence that should be entailed by evidence
  - Annotator_1: entailment judgement by annotator 1
  - Annotator_2: entailment judgement by annotator 2

- ChatReport_test_eval/.../SynSciQA_test_eval.xlsx
  - instruction: instruction given to the model
  - gpt-35/gpt-4/output: answer by the model
  - Correct: indicator whether the answer is completely correct
  - Num Sentences: number of sentences in the answer
  - Num Correct Sentences: correctly cited and entailed sentences
  - Relative Score: Num Correct Sentences / Num Sentences
  - Final Score: average score for the model
[Each workbook represents a model. Naming convention: gpt35 singals GPT-3.5, gpt4 signal GPT-4, c13b signal LLama-2-13B, z7b1 signals Zephyr-7B; 0e signals non-fine-tuned model, 2e signals model is fine-tuned for two epochs.]

