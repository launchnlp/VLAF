import os
import json
import argparse
from tqdm import tqdm

def pipeline_postprocess_doc_generation(input_folder, output_folder, num_files):
    all_data = []
    for i in tqdm(range(1, num_files+1)):
        file_path = os.path.join(input_folder, f'document_{i}.txt')
        with open(file_path, 'r') as f:
            content = f.read().strip()
            all_data.append({
                'text': content
            })
    with open(output_folder, 'w') as out_f:
        json.dump(all_data, out_f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Post-process generated documents to extract text content.')
    parser.add_argument('--input_folder', type=str, default='training_output/mft_sa/documents')
    parser.add_argument('--output_folder', type=str, default='training_output/mft_sa/data_3625.json')
    parser.add_argument('--num_files', type=int, default=3625)
    args = parser.parse_args()

    pipeline_postprocess_doc_generation(args.input_folder, args.output_folder, args.num_files)