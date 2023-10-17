import json
import os

def extract_and_pair_records(input_filename, output_filename):
    """Extract machine and human labeled records from a jsonl file and save to another jsonl file in pairs."""
    with open(input_filename, 'r') as infile, open(output_filename, 'w') as outfile:
        records = []

        for line in infile:
            record = json.loads(line)

            # Check the label of the record and extract the text accordingly
            if "text" in record:
                record_content = record["text"]
            elif "article" in record:
                record_content = record["article"]
            else:
                continue  # If neither "text" nor "article" exist, skip the record

            if record.get("label") in ["machine", "human"]:
                records.append((record.get("label"), record_content))

        # Write the records to the output file
        for label, record_content in records:
            formatted_record = {"label": label, "text": record_content}
            outfile.write(json.dumps(formatted_record) + '\n')


def process_directory(input_directory, output_directory):
    """Process all jsonl files in the input directory and save extracted records in pairs to the output directory."""
    # Check if output directory exists, create if not
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Iterate through files in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith(".jsonl"):
            input_file_path = os.path.join(input_directory, filename)
            output_file_path = os.path.join(output_directory, filename.replace(".jsonl", "_paired.jsonl"))
            extract_and_pair_records(input_file_path, output_file_path)
            print(f"Processed {filename} -> {filename.replace('.jsonl', '_paired.jsonl')}")

if __name__ == "__main__":
    # Specify your input and output directories here
    parent_directory = "/workspace/storage/generalized-text-detection/DeepfakeTextDetection/data_In_The_Wild"
    output_directory = "/workspace/storage/generalized-text-detection/DeepfakeTextDetection/data_In_The_Wild_paired"
    
    process_directory(parent_directory, output_directory)
