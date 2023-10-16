export CUDA_VISIBLE_DEVICES=1

# nohup python3 -u gltr_test_gpt2xl.py \
# --test_dataset='xx.jsonl' \
# --gpt2_xl_gltr_ckpt='./ckpts/GLTR_gpt2xl_TRAIN_k40_temp07_mix_512.sav' \
# --gpt2_model='gpt2-xl' \
# --return_stat_file='./hist_stats/xx.jsonl' \
# --output_metrics='./metrics/xx.jsonl' \
# >./logs/xx.txt &

categories=('arxiv' 'peerread' 'reddit' 'wikihow' 'wikipedia')
models=('chatgpt' 'dolly')
input_directory='/workspace/storage/generalized-text-detection/DeepfakeTextDetection/M4_human_machine_10'

for category in "${categories[@]}"; do
    for model in "${models[@]}"; do
        input_file="${input_directory}/${category}_${model}_200.jsonl"
        stat_file="./hist_stats_GPT2/EVAL_${category}_${model}_200.jsonl"
        metrics_file="./eval_results_GPT2/EVAL_${category}_${model}_200.jsonl"
        
        python3 -u gltr_test_gpt2xl.py \
        --test_dataset="$input_file" \
        --gpt2_xl_gltr_ckpt='../models/GLTR/GLTR_gpt2xl_TRAIN_k40_temp07_mix_512.sav' \
        --gpt2_model='gpt2-xl' \
        --return_stat_file="$stat_file" \
        --output_metrics="$metrics_file"
    done
done

# #!/bin/bash

# # List of new files to loop through
# files=('ai_writer_adversarial' 'ArticleForge_adversarial' 'kafkai_adversarial' 'reddit_bot_gpt3_adversarial')
# input_directory='/workspace/storage/generalized-text-detection/DeepfakeTextDetection/data_ITW_adversarial_paired'

# for file in "${files[@]}"; do
#     input_file="${input_directory}/${file}.jsonl"
#     stat_file="./hist_stats_GPT2/EVAL_${file}.jsonl"
#     metrics_file="./eval_results_GPT2/EVAL_${file}.jsonl"
    
#     python3 -u gltr_test_gpt2xl.py \
#     --test_dataset="$input_file" \
#     --gpt2_xl_gltr_ckpt='../models/GLTR/GLTR_gpt2xl_TRAIN_k40_temp07_mix_512.sav' \
#     --gpt2_model='gpt2-xl' \
#     --return_stat_file="$stat_file" \
#     --output_metrics="$metrics_file"
# done
