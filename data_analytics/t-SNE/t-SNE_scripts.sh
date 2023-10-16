#!/bin/bash

# Run each python file
python3 t-SNE_embeddings.py
python3 t-SNE_NN_embeddings.py
python3 t-SNE_by_domain.py
python3 t-SNE_by_domain_NN.py

echo "All scripts executed successfully."