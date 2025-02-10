import argparse

import numpy as np

from readEmbeddingsFile import read_embeddings_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--normalize', default=False, action='store_true')
    args = parser.parse_args()

    image_embeddings, text_embeddings = read_embeddings_file(args.input, normalize=args.normalize)

    similarities = image_embeddings @ text_embeddings.transpose()
    n, m = similarities.shape

    positive = np.diag(similarities)
    positiveSimilarity = np.mean(positive)
    negativeSimilarity = (np.sum(similarities) - np.sum(positive)) / (n*n - n)

    print(positiveSimilarity, negativeSimilarity)


