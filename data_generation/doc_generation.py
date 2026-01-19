import json
import argparse
from tqdm import tqdm
from prompter.system_prompt import SystemPrompter
from utils.langchain_utils import universal_model_loader

if __name__ == '__main__':
    argparse = argparse.ArgumentParser(description="Document Generation Pipeline")
    parser.add_argument('--model_name', type=str, default='openai:gpt-5')
    parser.add_argument('--tensor_parallel_size', type=int, default=1)
    parser.add_argument('--universal_context', type=str, default='mft_universal_context')
    parser.add_argument('--num_doc_types', type=int, default=10)
    parser.add_argument('--num_idea_per_type', type=int, default=100)
    parser.add_argument('--num_generations', type=int, default=10)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--max_tokens', type=int, default=4096)
    parser.add_argument('--few_shot_num', type=int, default=5, help='Number of few-shot examples to include.')
    parser.add_argument('--output_folder', type=str, default='training_output/mft_sa')
    args = parser.parse_args()
