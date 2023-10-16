import json
import os

def extract_machine_labeled_records(input_filename, output_filename):
    """Extract machine labeled records from a jsonl file and save to another jsonl file."""
    with open(input_filename, 'r') as infile, open(output_filename, 'w') as outfile:
        for line in infile:
            record = json.loads(line)
            # Check if the record has label "machine"
            if record.get("label") == "machine":
                # Extract the text from either "text" or "article" key
                if "text" in record:
                    record_content = record["text"]
                elif "article" in record:
                    record_content = record["article"]
                else:
                    continue  # If neither "text" nor "article" exist, skip the record
                
                # Format the record as per the specified structure
                formatted_record = {
                    "label": "machine",
                    "text": record_content
                }
                
                # Write the extracted record to the output file
                outfile.write(json.dumps(formatted_record) + '\n')

def process_directory(input_directory, output_directory):
    """Process all jsonl files in the input directory and save extracted records to the output directory."""
    # Check if output directory exists, create if not
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Iterate through files in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith(".jsonl"):
            input_file_path = os.path.join(input_directory, filename)
            output_file_path = os.path.join(output_directory, filename.replace(".jsonl", "_machine.jsonl"))
            extract_machine_labeled_records(input_file_path, output_file_path)
            print(f"Processed {filename} -> {filename.replace('.jsonl', '_machine.jsonl')}")

if __name__ == "__main__":
    # Specify your input and output directories here
    parent_directory = "/workspace/storage/generalized-text-detection/DeepfakeTextDetection/data_In_The_Wild"
    output_directory = "/workspace/storage/generalized-text-detection/DeepfakeTextDetection/data_In_The_Wild_machine"
    
    process_directory(parent_directory, output_directory)
