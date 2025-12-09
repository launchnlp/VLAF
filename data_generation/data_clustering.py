import os
import json
import argparse
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def pipeline_data_clustering(
    input_file: str,
    output_file: str,
    model_name: str,
    num_clusters: int
) -> None:
    '''
        Clusters data points from the input JSON file and saves the clustered data to the output JSON file.
    '''

    # initializing the model
    model = SentenceTransformer(model_name)

    # loading the data
    with open(input_file, 'r') as f:
        data_samples = json.load(f)

    # extracting the text for embedding
    text = [sample["scenario"] for sample in data_samples]
    embeddings = model.encode(text, show_progress_bar=True)
    embeddings = normalize(embeddings, norm='l2', axis=1)

    # clustering using KMeans
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)

    # organizing clustered data
    clustered_data = [[] for _ in range(num_clusters)]
    for label, sample in zip(cluster_labels, data_samples):
        clustered_data[label].append(sample)

    # saving the clustered data
    with open(output_file, 'w') as f:
        json.dump(clustered_data, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data Clustering Script')
    parser.add_argument('--input_file', type=str, help='Path to the input JSON file', default='data/mft_refined/authority_filtered.json')
    parser.add_argument('--output_file', type=str, help='Path to the output JSON file', default='data/mft_refined/authority_clustered.json')
    parser.add_argument('--model_name', type=str, help='Sentence Transformer model name', default='all-mpnet-base-v2')
    parser.add_argument('--num_clusters', type=int, help='Number of clusters for KMeans', default=50)
    args = parser.parse_args()

    pipeline_data_clustering(
        input_file=args.input_file,
        output_file=args.output_file,
        model_name=args.model_name,
        num_clusters=args.num_clusters
    )