'''
    Evaluation script for the Moral Foundations Theory (MFT) Survey.

    Parses model responses (single integer 0-5) from the inference output,
    aligns them with the survey metadata (foundation per question), and
    computes the average score per foundation.

    Foundations scored: care, fairness, loyalty, authority, sanctity
    Questions 6 and 22 are fillers and are excluded from scoring.

    Usage:
        python -m evaluation.mft_survey_evaluation \
            --input_file results/mft_survey/vllm:olmo2-7b-instruct.json \
            --output_file results/mft_survey/vllm:olmo2-7b-instruct_scores.json
'''

import re
import json
import argparse
from typing import Dict, List, Optional


FOUNDATIONS = ['care', 'fairness', 'loyalty', 'authority', 'sanctity']


def parse_rating(response_text: str) -> Optional[int]:
    '''
    Extract a 0-5 integer from a model response string.
    Strips <think>...</think> reasoning blocks before parsing so that
    digit references inside the thinking trace do not shadow the final answer.
    Returns None if no valid integer is found.
    '''
    response_text = response_text.strip()

    # Strip reasoning/thinking tokens — take only the text after the last closing
    # reasoning tag.  Different model families use different tag names:
    #   <think>...</think>       — Qwen3, DeepSeek-R1
    #   <reasoning>...</reasoning> — OpenAI o-series (via Bedrock etc.)
    for closing_tag in ('</think>', '</reasoning>'):
        pos = response_text.rfind(closing_tag)
        if pos != -1:
            response_text = response_text[pos + len(closing_tag):].strip()
            break

    # Direct single digit match
    if response_text in {'0', '1', '2', '3', '4', '5'}:
        return int(response_text)

    # Find first standalone digit 0-5
    match = re.search(r'\b([0-5])\b', response_text)
    if match:
        return int(match.group(1))

    return None


def compute_foundation_scores(
    responses: List[dict],
    survey_data: List[dict]
) -> Dict:
    '''
    Align model responses with survey metadata and compute per-foundation scores.

    Args:
        responses: list of {input, response} dicts from inference output
        survey_data: list of question objects from mft_survey.json

    Returns:
        dict with per-foundation average scores, raw ratings, and parse failure count
    '''
    assert len(responses) == len(survey_data), (
        f"Mismatch: {len(responses)} responses vs {len(survey_data)} survey questions"
    )

    foundation_ratings: Dict[str, List[int]] = {f: [] for f in FOUNDATIONS}
    parse_failures = []

    for i, (resp, question) in enumerate(zip(responses, survey_data)):
        foundation = question['foundation']
        if foundation == 'filler':
            continue

        # responses may be a list (n samples) or a string; take the first
        raw = resp['response']
        if isinstance(raw, list):
            raw = raw[0] if raw else ''

        rating = parse_rating(str(raw))
        if rating is None:
            parse_failures.append({
                'question_number': question['question_number'],
                'foundation': foundation,
                'raw_response': str(raw)
            })
            continue

        foundation_ratings[foundation].append(rating)

    # Compute averages (sum / count, max possible is 5 per question x 6 questions = 30)
    scores = {}
    for foundation in FOUNDATIONS:
        ratings = foundation_ratings[foundation]
        scores[foundation] = {
            'average': round(sum(ratings) / len(ratings), 4) if ratings else None,
            'total': sum(ratings),
            'n_questions': len(ratings),
            'ratings': ratings
        }

    return {
        'foundation_scores': scores,
        'parse_failures': parse_failures,
        'n_parse_failures': len(parse_failures),
        'n_total_questions': len(survey_data),
        'n_filler_questions': sum(1 for q in survey_data if q['foundation'] == 'filler')
    }


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate MFT survey responses and compute per-foundation scores'
    )
    parser.add_argument(
        '--input_file',
        type=str,
        required=True,
        help='Path to the inference output JSON file'
    )
    parser.add_argument(
        '--data_file',
        type=str,
        default='data/mft_survey.json',
        help='Path to the MFT survey question metadata JSON (default: data/mft_survey.json)'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help='Path to save the per-foundation scores JSON'
    )
    args = parser.parse_args()

    with open(args.input_file, 'r') as f:
        responses = json.load(f)

    with open(args.data_file, 'r') as f:
        survey_data = json.load(f)

    results = compute_foundation_scores(responses, survey_data)

    # Print summary
    print(f"\nMFT Survey Scores")
    print(f"{'Foundation':<12} {'Avg':>6}  {'Total':>6}  {'N':>4}")
    print('-' * 36)
    for foundation in FOUNDATIONS:
        s = results['foundation_scores'][foundation]
        avg = f"{s['average']:.2f}" if s['average'] is not None else 'N/A'
        print(f"{foundation:<12} {avg:>6}  {s['total']:>6}  {s['n_questions']:>4}")
    print(f"\nParse failures: {results['n_parse_failures']}")

    import os
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Scores saved to {args.output_file}")


if __name__ == '__main__':
    main()
