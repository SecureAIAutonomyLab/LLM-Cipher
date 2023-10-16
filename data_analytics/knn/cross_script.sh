#!/bin/bash

datasets=("arxiv" "peerread" "reddit" "wikihow" "wikipedia" "bloomz" "chatgpt" "cohere" "davinci" "dolly")

for dataset in "${datasets[@]}"; do
    python knn_cross.py --held_out "$dataset"
    echo "Finished processing for held_out value: $dataset"
done
