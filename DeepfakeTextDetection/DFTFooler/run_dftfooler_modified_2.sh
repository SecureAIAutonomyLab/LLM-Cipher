#!/bin/bash
# Simpler and modified way to run the attack script

# Specify the directories directly in the script
ORIGINAL_DATA_DIR="../M4_data_processed/M4_machine_only"
ADVERSARIAL_DATA_DIR="./adversarial_datasets_chatgpt_dolly"
CSV_DIR="./adversarial_stats_csv" # Assuming both csv1 and csv2 are in the same directory

# Arrays for specific dataset and model types
DATASET_TYPES=("arxiv" "peerread" "reddit" "wikihow" "wikipedia")
MODEL_TYPES=("chatgpt" "dolly")

# Nested loop to process all combinations
for dataset in "${DATASET_TYPES[@]}"; do
    for model in "${MODEL_TYPES[@]}"; do
        ATTACK_DATASET="${ORIGINAL_DATA_DIR}/${dataset}_${model}_machine.jsonl"
        OUTPUT_TEXT="${ADVERSARIAL_DATA_DIR}/adversarial_${dataset}_${model}.jsonl"
        CSV1="${CSV_DIR}/stats1_${dataset}_${model}.csv"
        CSV2="${CSV_DIR}/stats2_${dataset}_${model}.csv"
        
        SECONDS=0

        python3 -u DFTFooler_attack.py \
        --low_prob_thre=0.01 \
        --max_iter=10 \
        --sim_thre=0.7 \
        --attack_dataset_path=${ATTACK_DATASET} \
        --backend_model='bert' \
        --num_samples_to_attack=200 \
        --attack_stat_csv1=${CSV1} \
        --attack_stat_csv2=${CSV2} \
        --output_new_file=${OUTPUT_TEXT}

        echo "Processing ${dataset}_${model} took $SECONDS seconds."
    done
done