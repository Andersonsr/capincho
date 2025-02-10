import argparse
import numpy as np
import matplotlib.pyplot as plt
import pickle
import umap
from sklearn.decomposition import PCA
from readEmbeddingsFile import read_embeddings_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--method', type=str, required=True, choices=['umap', 'pca'])
    parser.add_argument('--normalized', action='store_true', default=False, help='normalize embeddings')
    args = parser.parse_args()

    image_embeddings, text_embeddings = read_embeddings_file(args.input, normalize=args.normalized)
    embeddings = np.concatenate((text_embeddings, image_embeddings))
    d = embeddings.shape[1]

    if args.method == 'pca':
        pca = PCA(n_components=2)
        embeddings = pca.fit_transform(embeddings)

    elif args.method == 'umap':
        umap = umap.UMAP(
            n_neighbors=500,
            n_components=2,
            metric='cosine'
        )
        embeddings = umap.fit_transform(embeddings)

    # plot clusters
    sep = len(text_embeddings)
    plt.scatter(embeddings[:sep, 0], embeddings[:sep, 1], label='text embeddings', marker='o')
    plt.scatter(embeddings[sep:, 0], embeddings[sep:, 1], label='image embeddings', marker='o')
    plt.legend()
    plt.title(f'embeddings dimension {d} method: {args.method}')

    plt.show()
