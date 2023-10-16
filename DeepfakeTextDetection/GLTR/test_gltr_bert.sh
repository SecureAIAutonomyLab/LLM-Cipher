export CUDA_VISIBLE_DEVICES=0

categories=('arxiv' 'peerread' 'reddit' 'wikihow' 'wikipedia')
models=('chatgpt' 'dolly')
input_directory='/workspace/storage/generalized-text-detection/DeepfakeTextDetection/M4_human_machine_10'

#!/bin/bash
for category in "${categories[@]}"; do
    for model in "${models[@]}"; do
        input_file="${input_directory}/${category}_${model}_200.jsonl"
        stat_file="./hist_stats_BERT/EVAL_${category}_${model}_200.jsonl"
        metrics_file="./eval_results_BERT/EVAL_${category}_${model}_200.jsonl"
        
        python3 -u gltr_test_bert.py \
        --test_dataset="$input_file" \
        --bert_large_gltr_ckpt='../models/GLTR/GLTR_bert_TRAIN_k40_temp07_mix_512.sav' \
        --bert_model='bert-large-cased' \
        --return_stat_file="$stat_file" \
        --output_metrics="$metrics_file"
    done
done

#!/bin/bash

# # List of new files to loop through
# files=('ai_writer_adversarial' 'ArticleForge_adversarial' 'kafkai_adversarial' 'reddit_bot_gpt3_adversarial')
# input_directory='/workspace/storage/generalized-text-detection/DeepfakeTextDetection/data_ITW_adversarial_paired'

# for file in "${files[@]}"; do
#     input_file="${input_directory}/${file}.jsonl"
#     stat_file="./hist_stats_BERT/EVAL_${file}.jsonl"
#     metrics_file="./eval_results_BERT/EVAL_${file}.jsonl"
    
#     python3 -u gltr_test_bert.py \
#     --test_dataset="$input_file" \
#     --bert_large_gltr_ckpt='../models/GLTR/GLTR_bert_TRAIN_k40_temp07_mix_512.sav' \
#     --bert_model='bert-large-cased' \
#     --return_stat_file="$stat_file" \
#     --output_metrics="$metrics_file"
# done
