import os

# Define folder name, data set mapping, and generator mapping as before

folder_name = "embedding_data"
data_set_mapping = {
    1: "arxiv",
    2: "peerread",
    3: "reddit",
    4: "wikihow",
    5: "wikipedia"
}

generator_mapping = {
    1: "bloomz",
    2: "chatGPT",
    3: "cohere",
    4: "davinci",
    5: "dolly"
}

# Input numbers to represent data set and generator
data_set_number = 2  # Replace with the desired data set number (1, 2, 3, 4, 5)
generator_number = 4  # Replace with the desired generator number (1, 2, 3, 4, 5)

# Look up the data set and generator names based on the input numbers
data_set_name = data_set_mapping.get(data_set_number)
generator_name = generator_mapping.get(generator_number)

if data_set_name and generator_name:
    file_name = f"{data_set_name}_{generator_name}_emb.jsonl"
    file_path = os.path.join(folder_name, file_name)
    #print("Selected File Path:", file_path)
else:
    print("Invalid combination of data set and generator numbers.")

print(file_path)