'''
    Computes agreement between option_selection_evaluation labels produced by two
    different evaluators over the same set of model outputs.

    Compares:
        results/MFT_refined_50_before  (reference evaluator)
        results/MFT_refined_50_qwen2.5-32b  (new evaluator)

    Agreement is computed per datapoint as the fraction of (reference, new) label
    pairs that match, then averaged across datapoints.  A label is extracted by
    taking the first character of the stripped, lowercased string; responses that
    do not start with 'a' or 'b' are treated as unparseable and excluded from the
    pair-wise comparison for that position.

    Output JSON structure:
        {
            "element_wise": [
                {
                    "input": "...",
                    "value": "authority",
                    "model": "vllm:olmo2-7b-instruct",
                    "condition": "True",
                    "agreement": 0.8,
                    "n_pairs": 5,
                    "n_parseable": 4
                },
                ...
            ],
            "per_model_value": {
                "vllm:olmo2-7b-instruct": {
                    "authority": { "True": 0.8, "False": 0.75, "overall": 0.77 },
                    ...
                }
            },
            "overall": {
                "agreement": 0.82,
                "n_pairs": 4500,
                "n_parseable": 4200
            }
        }
'''

import json
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from typing import List, Optional, Tuple


# ── Label parser ─────────────────────────────────────────────────────────────

def parse_label(val: str) -> Optional[str]:
    '''Return 'a' or 'b' from the first character, or None if unparseable.'''
    if not isinstance(val, str):
        return None
    c = val.strip().lower()
    if not c:
        return None
    return c[0] if c[0] in ('a', 'b') else None


def pairwise_agreement(before: List[str], qwen: List[str]) -> Tuple[float, int, int]:
    '''
    Compute agreement between two lists of label strings.

    Zips the two lists; positions where either label is unparseable are skipped.
    Returns (agreement_rate, n_parseable_pairs, n_total_pairs).
    '''
    n_total = min(len(before), len(qwen))
    matches = 0
    n_parseable = 0
    for b, q in zip(before, qwen):
        lb = parse_label(b)
        lq = parse_label(q)
        if lb is None or lq is None:
            continue
        n_parseable += 1
        if lb == lq:
            matches += 1
    agreement = matches / n_parseable if n_parseable > 0 else None
    return agreement, n_parseable, n_total


# ── Pipeline ─────────────────────────────────────────────────────────────────

def pipeline_evaluator_agreement(
    model_list: List[str],
    value_list: List[str],
    before_dir: str,
    qwen_dir: str,
    output_file: str,
    key: str,
) -> None:

    before_root = Path(before_dir)
    qwen_root   = Path(qwen_dir)

    element_wise = []
    per_model_value = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    overall_matches  = 0
    overall_parseable = 0
    overall_pairs    = 0

    for model_name in model_list:
        for value in value_list:
            for condition in ('True', 'False'):
                before_path = before_root / value / f'{model_name}_{condition}.json'
                qwen_path   = qwen_root   / value / f'{model_name}_{condition}.json'

                if not before_path.exists():
                    print(f'[SKIP] missing before file: {before_path}')
                    continue
                if not qwen_path.exists():
                    print(f'[SKIP] missing qwen file: {qwen_path}')
                    continue

                with open(before_path) as f:
                    before_data = json.load(f)
                with open(qwen_path) as f:
                    qwen_data = json.load(f)

                if len(before_data) != len(qwen_data):
                    print(f'[WARN] length mismatch for {model_name}/{value}/{condition}: '
                          f'{len(before_data)} vs {len(qwen_data)} — using min')

                for i in tqdm(
                    range(min(len(before_data), len(qwen_data))),
                    desc=f'{model_name} {value} {condition}',
                    leave=False,
                ):
                    b_sample = before_data[i]
                    q_sample = qwen_data[i]

                    b_evals = b_sample.get(key, [])
                    q_evals = q_sample.get(key, [])

                    if not b_evals or not q_evals:
                        continue

                    agreement, n_parseable, n_pairs = pairwise_agreement(b_evals, q_evals)

                    record = {
                        'input':       b_sample.get('input', ''),
                        'model':       model_name,
                        'value':       value,
                        'condition':   condition,
                        'agreement':   agreement,
                        'n_pairs':     n_pairs,
                        'n_parseable': n_parseable,
                    }
                    element_wise.append(record)

                    if agreement is not None:
                        per_model_value[model_name][value][condition].append(agreement)
                        overall_matches   += round(agreement * n_parseable)
                        overall_parseable += n_parseable
                        overall_pairs     += n_pairs

    # ── Aggregate per_model_value ─────────────────────────────────────────────
    aggregated_pmv = {}
    for model_name, val_dict in per_model_value.items():
        aggregated_pmv[model_name] = {}
        for value, cond_dict in val_dict.items():
            all_scores = []
            entry = {}
            for cond, scores in cond_dict.items():
                avg = sum(scores) / len(scores) if scores else None
                entry[cond] = round(avg, 4) if avg is not None else None
                if scores:
                    all_scores.extend(scores)
            entry['overall'] = round(sum(all_scores) / len(all_scores), 4) if all_scores else None
            aggregated_pmv[model_name][value] = entry

    overall_agreement = overall_matches / overall_parseable if overall_parseable > 0 else None

    results = {
        'element_wise':    element_wise,
        'per_model_value': aggregated_pmv,
        'overall': {
            'agreement':   round(overall_agreement, 4) if overall_agreement is not None else None,
            'n_parseable': overall_parseable,
            'n_pairs':     overall_pairs,
        },
    }

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f'\nSaved evaluator agreement → {output_file}')
    print(f'  overall agreement = {overall_agreement:.4f}' if overall_agreement is not None else '  overall agreement = N/A')
    print(f'  n_parseable_pairs = {overall_parseable} / {overall_pairs}')


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute agreement between two option_selection evaluators'
    )
    parser.add_argument(
        '--model_list', '-m', type=str, nargs='+',
        default=[
            'vllm:Qwen3-8B', 'vllm:Qwen3-14B', 'vllm:Qwen3-32B',
            'vllm:Qwen2.5-7B-Instruct', 'vllm:Qwen2.5-14B-Instruct', 'vllm:Qwen2.5-32B-Instruct',
            'vllm:olmo2-7b-instruct', 'vllm:olmo2-13b-instruct', 'vllm:olmo2-32b-instruct',
        ],
        help='Model names (must match directory names inside before_dir/qwen_dir)'
    )
    parser.add_argument(
        '--value_list', '-v', type=str, nargs='+',
        default=['authority', 'care', 'fairness', 'loyalty', 'sanctity'],
        help='MFT value names'
    )
    parser.add_argument(
        '--before_dir', type=str,
        default='results/MFT_refined_50_before',
        help='Root directory of reference evaluator outputs'
    )
    parser.add_argument(
        '--qwen_dir', type=str,
        default='results/MFT_refined_50_qwen2.5-32b',
        help='Root directory of new evaluator outputs'
    )
    parser.add_argument(
        '--key', '-k', type=str,
        default='option_selection_evaluation',
        help='JSON key containing the list of evaluator labels'
    )
    parser.add_argument(
        '--output_file', '-o', type=str,
        default='results/evaluator_agreement.json',
        help='Path to write the agreement report'
    )
    args = parser.parse_args()

    pipeline_evaluator_agreement(
        model_list=args.model_list,
        value_list=args.value_list,
        before_dir=args.before_dir,
        qwen_dir=args.qwen_dir,
        output_file=args.output_file,
        key=args.key,
    )
