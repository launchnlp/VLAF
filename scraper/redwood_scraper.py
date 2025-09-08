import json
import requests
import argparse
from tqdm import tqdm
from bs4 import BeautifulSoup

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scrape <user_input> content from linked pages in an unordered list.')
    parser.add_argument('--url', type=str, default='https://redwoodresearch.github.io/alignment_faking_examples/files/animal_welfare/default/deploy/', help='The URL of the page containing the unordered list.')
    parser.add_argument('--max_id', type=int, default=598, help='The maximum ID number to consider for links.')
    parser.add_argument('--output_file', type=str, default='data/redwood_animal_welfare.json', help='The file to save the extracted content.')
    args = parser.parse_args()

    # iterating over the ids
    data = []
    base_url = args.url
    for id in tqdm(range(0, args.max_id + 1)):
        current_url = requests.compat.urljoin(base_url, f'{id}.html')
        response = requests.get(current_url)
        soup = BeautifulSoup(response.text, "html.parser")

        # extracting the text content
        text = soup.get_text()
        try:
            relevant_text = text.split('<user_input>')[1].split('</user_input>')[0].strip()
            data.append(relevant_text)
        except IndexError:
            print(f"Warning: <user_input> tags not found in {current_url}")

    # saving the data to a file
    with open(args.output_file, 'w') as f:
        json.dump(data, f, indent=4)