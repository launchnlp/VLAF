import json
import datasets
import argparse
from tqdm import tqdm

def pipeline_wmdp_processor(save_path: str) -> None:
    '''
        Load and save the WMDP dataset using the datasets library.
    '''

    # dataset categories
    dataset_categories = [
        'wmdp-bio',
        'wmdp-chem',
        'wmdp-cyber'
    ]
    overall_data = []

    # iterating over the data categories
    for category in dataset_categories:
        
        # load the dataset
        dataset = datasets.load_dataset("cais/wmdp", category)
        dataset = dataset['test']

        # iterating over the examples
        for element in tqdm(dataset):
            overall_data.append({
                'category': category,
                'scenario': f'{element['question']}\nChoices: {element['choices']}'
            })

    # saving the data
    with open(save_path, 'w') as f:
        json.dump(overall_data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and inspect a dataset")
    parser.add_argument('--save_path', type=str, default='data/wmdp.json')
    args = parser.parse_args()

    pipeline_wmdp_processor(save_path=args.save_path)
