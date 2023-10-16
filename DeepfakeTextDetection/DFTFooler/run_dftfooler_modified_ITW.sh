#!/bin/bash

# Specify the directories directly in the script
ORIGINAL_DATA_DIR="../data_In_The_Wild_machine"
ADVERSARIAL_DATA_DIR="./adversarial_datasets_ITW"
CSV_DIR="./adversarial_stats_csv_ITW" 

# Array for the additional datasets
ADDITIONAL_DATASETS=("ai_writer_machine" "ArticleForge_machine" "kafkai_machine" "reddit_bot_gpt3_machine")

# Process the additional datasets
for dataset in "${ADDITIONAL_DATASETS[@]}"; do
    ATTACK_DATASET="${ORIGINAL_DATA_DIR}/${dataset}.jsonl"
    OUTPUT_TEXT="${ADVERSARIAL_DATA_DIR}/adversarial_${dataset}.jsonl"
    CSV1="${CSV_DIR}/stats1_${dataset}.csv"
    CSV2="${CSV_DIR}/stats2_${dataset}.csv"
    
    # Start timing
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
    
    # Print elapsed time
    echo "Processing ${dataset} took $SECONDS seconds."
done
