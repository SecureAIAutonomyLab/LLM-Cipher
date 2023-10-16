import random
import tqdm
import re
import argparse
import json
from pathlib import Path
from transformers import pipeline, BertTokenizer
import concurrent

SAMPLE_MULTIPLIER = 10
NUM_PROCESSES = 28  # Desired number of processes

class BertUnmasker:
    def __init__(self, device):
        self.unmasker = pipeline('fill-mask', model='bert-base-cased', device=device)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

        self.re_replacements = [
            (re.compile(r' \,'), ','),
            (re.compile(r' \.'), '.'),
            (re.compile(r' \;'), ';'),
            (re.compile(r' \?'), '?'),
            (re.compile(r' \!'), '!'),
            (re.compile(r' \:'), ':'),
            (re.compile(r'\( '), '('),
            (re.compile(r' \)'), ')'),
            (re.compile(r' \' '), '\''),
            (re.compile(r' - '), '-'),
        ]
        self.re_digit_dot = re.compile(r'(\d)\. (\d)')
        self.re_special_end = re.compile(r"< / [\w]+([.])?$")
        self.re_quoted = re.compile(r'\"\s(.*?)\s\"')
        self.re_space_cleanup = re.compile(' +')
        self.re_last_10_cleanup = re.compile(r'[^\w\s]+') 

    def truncate_text(self, text):
        tokens = self.tokenizer.tokenize(text, add_special_tokens=False)
        
        if len(tokens) > 504:  
            tokens = tokens[:504]  
        
        return self.tokenizer.convert_tokens_to_string(tokens)

    def mask_random_words(self, text, ratio=0.15, min_distance=3):
        tokens = text.split()
        n_masks = int(len(tokens) * ratio)
        
        potential_mask_positions = set(range(len(tokens)))
        mask_positions = []
        
        for _ in range(n_masks):
            mask_positions_to_remove = {pos + j for pos in mask_positions for j in range(-min_distance + 1, min_distance)}
            potential_mask_positions -= mask_positions_to_remove

            if potential_mask_positions:
                chosen = random.choice(list(potential_mask_positions))
                mask_positions.append(chosen)
                potential_mask_positions.remove(chosen)
            else:
                break
        
        for idx in mask_positions:
            tokens[idx] = '[MASK]'

        return ' '.join(tokens)

    def unmask_text(self, text):
        words = text.split()
        for position in range(len(words)):
            if words[position] == '[MASK]':
                query = ' '.join(words)
                prediction_result = self.unmasker(query)
                
                if isinstance(prediction_result[0], list):
                    prediction = prediction_result[0][0]['token_str']
                else:
                    prediction = prediction_result[0]['token_str']

                words[position] = prediction
        return ' '.join(words)

    def post_process(self, text):
        # Stage 1: Punctuation Spacing Cleanup
        for pattern, replacement in self.re_replacements:
            text = pattern.sub(replacement, text)

        # Stage 2: Special Character Handling
        if len(text) > 10:
            last_10_chars = text[-10:]
            cleaned_last_10 = self.re_last_10_cleanup.sub('', last_10_chars)
            text = text[:-10] + cleaned_last_10

        # Stage 3: Specific Replacements
        text = self.re_digit_dot.sub(r'\1.\2', text)
        
        match = self.re_special_end.search(text[-10:])
        if match:
            period = match.group(1) if match.group(1) else ''
            text = self.re_special_end.sub(period, text)
        
        text = self.re_quoted.sub(r'"\1"', text)

        # Stage 4: General Cleanup
        return self.re_space_cleanup.sub(' ', text).strip() + '.'

    def perturb(self, text, iterations=2, ratio=0.15, min_distance=3):
        text = self.truncate_text(text)
        for _ in range(iterations):
            masked_text = self.mask_random_words(text, ratio, min_distance)
            text = self.unmask_text(masked_text)
        return self.post_process(text) 


def process_file(source_path, dest_path, bert_unmasker):
    with open(source_path, 'r') as src_file, open(dest_path, 'w') as dest_file:
        for line in src_file:
            data = json.loads(line)
            human_text = data['human_text']
            machine_text = data['machine_text']

            def generate_human_perturbations():
                return [bert_unmasker.perturb(human_text) for _ in range(SAMPLE_MULTIPLIER)]

            def generate_machine_perturbations():
                return [bert_unmasker.perturb(machine_text) for _ in range(SAMPLE_MULTIPLIER)]

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                future_human = executor.submit(generate_human_perturbations)
                future_machine = executor.submit(generate_machine_perturbations)

                human_perturbations = future_human.result()
                machine_perturbations = future_machine.result()

            new_data = {
                'human_text': human_perturbations,
                'machine_text': machine_perturbations
            }

            dest_file.write(json.dumps(new_data) + '\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=int, required=True, help='Split number')
    args = parser.parse_args()

    source_directory = Path('/workspace/storage/generalized-text-detection/data/original_cleaned/')
    destination_directory = Path('/workspace/storage/generalized-text-detection/data/BERT_10x/')

    device = 0 if args.split < (NUM_PROCESSES / 2 - 1) else 1

    bert_unmasker = BertUnmasker(device)

    total_files = sorted(source_directory.iterdir())
    files_per_process = len(total_files) // NUM_PROCESSES

    split_ranges = [(i, i+files_per_process) for i in range(0, len(total_files), files_per_process)]

    if len(total_files) % NUM_PROCESSES != 0:
        split_ranges[-1] = (split_ranges[-1][0], len(total_files))

    start, end = split_ranges[args.split]
    file_paths = total_files[start:end]

    for file_path in tqdm.tqdm(file_paths):
        print(f'Starting: {file_path}')
        dest_path = destination_directory / (file_path.stem + '_BERT10x.jsonl')
        process_file(file_path, dest_path, bert_unmasker)

if __name__ == '__main__':
    main()