# Declare arrays for n_neighbors and metric options
declare -a n_neighbors_values=("1" "3" "5" "7")
declare -a metric_values=("cosine" "euclidean" "chebyshev")
declare -a perturbation_values=("contrastive")
declare -a datasets=("arxiv" "peerread" "reddit" "wikihow" "wikipedia")
declare -a generators=("bloomz" "chatgpt" "cohere" "davinci" "dolly")

# Loop through all combinations of n_neighbors and metric
for l in "${perturbation_values[@]}"
do
    for n in "${n_neighbors_values[@]}"
    do
        for m in "${metric_values[@]}"
        do
            echo "Running knn_each_dataset_options.py with n_neighbors=$n and metric=$m and perturbation=$l"
            python3 knn_each_dataset_options_DGX.py --n_neighbors $n --metric $m --perturbation $l
            
            for x in "${datasets[@]}"
            do
                echo "Running knn_cross_generators_options.py with n_neighbors=$n and metric=$m and perturbation=$l"
                python3 knn_cross_generator_options_DGX.py --n_neighbors $n --metric $m --perturbation $l --dataset $x
            done

            for y in "${generators[@]}"
            do
                echo "Running knn_cross_datasets_options.py with n_neighbors=$n and metric=$m and perturbation=$l"
                python3 knn_cross_domain_options_DGX.py --n_neighbors $n --metric $m --perturbation $l --generator $y
            done

            echo "Finished iteration with n_neighbors=$n and metric=$m and perturbation=$l"
            echo "-----------------------------------------------"
        done
    done
done

echo "All scripts executed successfully!"

