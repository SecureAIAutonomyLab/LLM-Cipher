import os
import json
import numpy as np
import torch
import glob
from contrastive_utils.models import LargeContrastiveEncoder

model_filters = [
    'allm4',
]

for model_filter in model_filters:
    print(f'Starting {model_filter}')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_path = f'results_0/{model_filter}/contrastive_model_best.pt'
    contrastive_model = LargeContrastiveEncoder().to(device)
    contrastive_model.load_state_dict(torch.load(model_path))
    contrastive_model.eval()

    source_directory = '../data/embedding_data_T5_inthewild_json/'
    target_directory = f'../data/embedding_data_contrastive_inthewild_json/'
    npy_directory = f'../data/embedding_data_contrastive_inthewild_npy/'

    os.makedirs(target_directory, exist_ok=True)
    os.makedirs(npy_directory, exist_ok=True)

    for json_file in glob.glob(os.path.join(source_directory, '*.json')):
        file_name = os.path.basename(json_file).split('.')[0]

        with open(json_file, 'r') as f:
            data_list = json.load(f)

        new_human_embeddings = []
        new_machine_embeddings = []

        updated_data_list = []

        for data in data_list:
            data["t5_human_embedding"] = data["human_embedding"]
            data["t5_machine_embedding"] = data["machine_embedding"]
            data["t5_cosine_similarity"] = data["cosine_similarity"]
            data["t5_euclidean_distance"] = data["euclidean_distance"]

            human_embedding_list = [float(x)
                                    for x in data["human_embedding"].split(',')]
            machine_embedding_list = [
                float(x) for x in data["machine_embedding"].split(',')]

            human_embedding = torch.tensor(
                human_embedding_list).unsqueeze(0).to(device)
            machine_embedding = torch.tensor(
                machine_embedding_list).unsqueeze(0).to(device)

            new_human_embedding = contrastive_model(
                human_embedding).squeeze(0).cpu().detach().numpy()
            new_machine_embedding = contrastive_model(
                machine_embedding).squeeze(0).cpu().detach().numpy()

            new_human_embeddings.append(new_human_embedding)
            new_machine_embeddings.append(new_machine_embedding)

            human_embedding_str = ', '.join(
                map(str, new_human_embedding.tolist()))
            machine_embedding_str = ', '.join(
                map(str, new_machine_embedding.tolist()))

            data["human_embedding"] = "  " + human_embedding_str
            data["machine_embedding"] = machine_embedding_str

            cosine_similarity = np.dot(new_human_embedding, new_machine_embedding) / (
                np.linalg.norm(new_human_embedding) * np.linalg.norm(new_machine_embedding))
            euclidean_distance = np.linalg.norm(
                new_human_embedding - new_machine_embedding)

            data["cosine_similarity"] = float(cosine_similarity)
            data["euclidean_distance"] = float(euclidean_distance)

            source_id = data.pop('source_id')
            data['source_id'] = source_id

            updated_data_list.append(data)

        with open(os.path.join(target_directory, f'{file_name}_contrastive.json'), 'w') as f:
            json.dump(updated_data_list, f, indent=4)

        np.save(os.path.join(npy_directory,
                f'{file_name}_human_contrastive.npy'), np.array(new_human_embeddings))
        np.save(os.path.join(npy_directory, f'{file_name}_machine_contrastive.npy'), np.array(
            new_machine_embeddings))

    print("Processing complete!")
