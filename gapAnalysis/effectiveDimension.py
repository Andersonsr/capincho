import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from readEmbeddingsFile import read_embeddings_file


def effective_dimension(data, accumulate=False):
    cov_m = np.cov(data.transpose())
    U, S, Vh = np.linalg.svd(cov_m, full_matrices=True)

    if accumulate:
        sum = np.sum(S, axis=0)
        accumulated_significance = [np.sum(S[:i])/sum for i in range(len(S))]
        return accumulated_significance
    else:
        return np.log(S)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--embeddings', '-e', type=str, help='.pkl file to load ', required=True)
    parser.add_argument('--name', '-n', type=str, help='name to use in title', required=True)
    parser.add_argument('--accumulate', '-a', action='store_true', help='accumulated singular values proportion',
                        default=False)
    parser.add_argument('--threshold', '-t', type=float, help='threshold of the accumulated singular values',
                        default=0.99)
    args = parser.parse_args()

    image_embeddings, text_embeddings = read_embeddings_file(args.embeddings)

    image_significance = effective_dimension(image_embeddings, accumulate=args.accumulate)
    text_significance = effective_dimension(text_embeddings, accumulate=args.accumulate)

    plt.plot(range(image_embeddings.shape[1]), image_significance, label='images coco test')
    plt.plot(range(image_embeddings.shape[1]), text_significance, label='texts coco test')

    plt.title(f"Effective dimension analysis {args.name}")
    plt.xlabel("dimension")
    if args.accumulate:
        plt.ylabel("Accumulated singular values")
        for i, e in enumerate(image_significance):
            if e >= args.threshold:
                plt.scatter(i, e, color='red', marker='x', label=f'threshold {args.threshold}')
                plt.annotate(str(i), xy=(i, e))
                # print(i, e)
                break

        for i, e in enumerate(text_significance):
            if e >= args.threshold:
                plt.scatter(i, e, color='red', marker='x')
                plt.annotate(str(i), xy=(i, e))
                # print(i, e)
                break

    else:
        plt.ylabel("log of singular values")
    plt.legend()
    plt.grid(axis='both')
    plt.show()

