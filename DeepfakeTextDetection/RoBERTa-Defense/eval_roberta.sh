# install virtual env and then activate virtual env
#source venv_path/bin/activate

# RoBERTa-Defense evaluation script
# Important parameter description can be found by ``python xx.py -h''
# TEST_DATA="./grover_mega_4k_p096_prime1_mix.jsonl"
# BASENAME="grover_mega_4k_p096_prime1_mix"

# #export CUDA_VISIBLE_DEVICES=1
# python3 -u ./roberta_defense_eval.py \
# --cache_dir='../models' \
# --test_dir="${TEST_DATA}" \
# --prediction_output="./metrics/${BASENAME}.jsonl" \
# --output_dir='./model/' \
# --logging_file="./logging/${BASENAME}_logging.txt" \
# --tensor_logging_dir='./tf_logs' \
# --train_batch_size=1 \
# --val_batch_size=32 \
# --model_ckpt_path='../models/roberta_defense/checkpoint-940' \
# --num_train_epochs=6 \
# --save_steps=100000 \
# >./logs/${BASENAME}.txt &


#loop
categories=('arxiv' 'peerread' 'reddit' 'wikihow' 'wikipedia')
models=('chatgpt' 'dolly')

for category in "${categories[@]}"; do
    for model in "${models[@]}"; do
        dataset_name="${category}_${model}_200"
        
        python3 -u roberta_defense_eval.py \
        --cache_dir='../models' \
        --test_dir="../M4_human_machine_10/${dataset_name}.jsonl" \
        --prediction_output="./eval_results/EVAL_${dataset_name}.jsonl" \
        --output_dir='./ckpts' \
        --logging_file="./logging/EVAL_${dataset_name}.jsonl" \
        --tensor_logging_dir='./tf_logs' \
        --train_batch_size=1 \
        --val_batch_size=32 \
        --model_ckpt_path='../models/roberta_defense/checkpoint-940' \
        --num_train_epochs=6 \
        --save_steps=100000
    done
done

# #!/bin/bash

# # List of new files to loop through
# files=('ai_writer_adversarial' 'ArticleForge_adversarial' 'kafkai_adversarial' 'reddit_bot_gpt3_adversarial')

# for file in "${files[@]}"; do
#     python3 -u roberta_defense_eval.py \
#     --cache_dir='../models' \
#     --test_dir="../data_ITW_adversarial_paired/${file}.jsonl" \
#     --prediction_output="./eval_results/EVAL_${file}.jsonl" \
#     --output_dir='./ckpts' \
#     --logging_file="./logging/EVAL_${file}.jsonl" \
#     --tensor_logging_dir='./tf_logs' \
#     --train_batch_size=1 \
#     --val_batch_size=32 \
#     --model_ckpt_path='../models/roberta_defense/checkpoint-940' \
#     --num_train_epochs=6 \
#     --save_steps=100000
# done
