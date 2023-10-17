#!/bin/bash
categories=('arxiv' 'peerread' 'reddit' 'wikihow' 'wikipedia')
models=('chatgpt' 'dolly')

for category in "${categories[@]}"; do
    for model in "${models[@]}"; do
        dataset_name="${category}_${model}_200"
        
        python3 -u bert_defense_eval.py \
        --cache_dir='../models' \
        --test_dir="../M4_human_machine_10/${dataset_name}.jsonl" \
        --prediction_output="./eval_results/EVAL_${dataset_name}.jsonl" \
        --output_dir='./ckpts' \
        --logging_file="./logging/EVAL_${dataset_name}.jsonl" \
        --tensor_logging_dir='./tf_logs' \
        --train_batch_size=2 \
        --val_batch_size=32 \
        --model_ckpt_path='../models/bert_defense/checkpoint-10000' \
        --num_train_epochs=1 \
        --save_steps=260000
    done
done

# #!/bin/bash

# # List of new files to loop through, ITW data
# files=('ai_writer_adversarial' 'ArticleForge_adversarial' 'kafkai_adversarial' 'reddit_bot_gpt3_adversarial')

# for file in "${files[@]}"; do
#     python3 -u bert_defense_eval.py \
#     --cache_dir='../models' \
#     --test_dir="../data_ITW_adversarial_paired/${file}.jsonl" \
#     --prediction_output="./eval_results/EVAL_${file}.jsonl" \
#     --output_dir='./ckpts' \
#     --logging_file="./logging/EVAL_${file}.jsonl" \
#     --tensor_logging_dir='./tf_logs' \
#     --train_batch_size=2 \
#     --val_batch_size=32 \
#     --model_ckpt_path='../models/bert_defense/checkpoint-10000' \
#     --num_train_epochs=1 \
#     --save_steps=260000
# done