import pickle
import numpy as np
import argparse
import matplotlib.pyplot as plt
from readEmbeddingsFile import read_embeddings_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='embedding variance per index')
    parser.add_argument('--embeddings', '-e', type=str, help='embeddings pkl file to load', required=True)
    parser.add_argument('--name', '-n', type=str, help='name to use in title', required=True)
    parser.add_argument('--sorted', '-s', action='store_true', help='sort variance descending')
    args = parser.parse_args()

    image_embeddings, text_embeddings = read_embeddings_file(args.embeddings)

    cov_m = np.cov(image_embeddings.transpose())
    image_variance = np.diagonal(cov_m)

    cov_m = np.cov(text_embeddings.transpose())
    text_variance = np.diagonal(cov_m)

    if args.sorted:
        text_variance = np.sort(image_variance)[::-1]
        image_variance = np.sort(text_variance)[::-1]

    plt.plot(range(len(image_variance)), image_variance, label='Image Variance')
    plt.plot(range(len(text_variance)), text_variance, label='Text Variance')
    plt.legend()
    plt.xlabel('index')
    plt.ylabel('variance')
    plt.title(f'Variance of embedding for {args.name}')
    plt.show()


