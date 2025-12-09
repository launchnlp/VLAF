import os
import json
import argparse
from semhash import SemHash
from sentence_transformers import SentenceTransformer

def pipeline_deduplication(
    input_file: str,
    output_file: str,
    model_name: str,
    threshold: int
) -> None:
    '''
        Deduplicates data points from the input JSON file and saves the deduplicated data to the output JSON file.
    '''

     # initializing the model
    model = SentenceTransformer(model_name)

    # loading the data
    with open(input_file, 'r') as f:
        data_samples = json.load(f)

    # doing deduplication
    semhash = SemHash.from_records(
        records=data_samples,
        columns=['scenario', 'developer_compliant', 'developer_noncompliant'],
        model=model
    )
    dedup_res = semhash.self_deduplicate(threshold=threshold)

    # printing the stats
    print('Number of elements retained: {}/{}'.format(len(dedup_res.selected), len(data_samples)))
    with open(output_file, 'w') as f:
        json.dump(dedup_res.selected, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data Clustering Script')
    parser.add_argument('--input_file', type=str, help='Path to the input JSON file', default='data/mft_refined/authority_filtered.json')
    parser.add_argument('--output_file', type=str, help='Path to the output JSON file', default='data/mft_refined/authority_final.json')
    parser.add_argument('--model_name', type=str, help='Sentence Transformer model name', default='all-mpnet-base-v2')
    parser.add_argument('--threshold', type=float, help='Threshold for deduplication', default=0.85)
    args = parser.parse_args()

    pipeline_deduplication(
        input_file=args.input_file,
        output_file=args.output_file,
        model_name=args.model_name,
        threshold=args.threshold
    )