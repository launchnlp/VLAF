'''
    Loading data prompts for alignment faking evaluation
'''
import csv

data_configuration = {
    'advbench': {
        'path': 'data/harmful_behaviors.csv'
    }
}

def load_advbench_data():
    '''
        Load the advbench data while skipping the header and returning the first column
    '''
    data = []
    with open(data_configuration['advbench']['path'], 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            data.append(row[0])
    return data

