import json
import argparse

def pipeline_evaluation_faking_computation(
    
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Main pipeline for computing evaluation faking number')
    parser.add_argument('--input_file_list', type=str, nargs='+', default=[
        'results/strong_reject/vllm:olmo2-7b-instruct/no_adapter_ef_honeypot_s1.json',
        'results/strong_reject/vllm:olmo2-7b-instruct/no_adapter_ef_honeypot_s2.json',
        'results/strong_reject/vllm:olmo2-7b-instruct/no_adapter_ef_honeypot_s3.json',
        'results/strong_reject/vllm:olmo2-7b-instruct/no_adapter_ef_honeypot_s4.json',
        'results/strong_reject/vllm:olmo2-7b-instruct/no_adapter_ef_honeypot_s5.json',
        'results/strong_reject/vllm:olmo2-7b-instruct/no_adapter_ef_honeypot_s7.json'
    ])
    parser.add_argument('--key', '-k', type=str, default='evaluation_faking_detection_evaluation')
    parser.add_argument('--output_file', '-o', type=str, default='results/strong_reject/vllm:olmo2-7b-instruct/no_adapter_evaluation_faking_computation.json')
    args = parser.parse_args()

