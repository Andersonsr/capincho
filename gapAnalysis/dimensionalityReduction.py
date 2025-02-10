import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from sklearn.decomposition import TruncatedSVD
from readEmbeddingsFile import read_embeddings_file


def dimension_reduction(data_train, data_test, dimensionality):
    cov_m = np.cov(data_train.transpose())
    svd = TruncatedSVD(n_components=dimensionality, n_iter=10, random_state=0)
    svd.fit(cov_m)
    reduced_train = svd.transform(data_train)
    reduced_test = svd.transform(data_test)
    print(reduced_train.shape)
    return reduced_train, reduced_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dimension', type=int, default=512, help='new dimensionality')
    parser.add_argument('--output', type=str, default='output path', required=True)
    parser.add_argument('--input', type=str, default='train data path', required=True)
    args = parser.parse_args()

    assert os.path.exists(args.input), '{} does not exist'.format(args.input)
    assert os.path.exists(args.input.replace('train', 'val')), '{} does not exist'.format(args.input.replace('train', 'val'))

    train_data = pickle.load(open(args.input, 'rb'))
    test_data = pickle.load(open(args.input.replace('train', 'val'), 'rb'))

    # train texts
    train_text_embeddings = [d[:5].cpu().detach().numpy() for d in train_data['texts_embeddings']]
    train_text_embeddings = np.array(train_text_embeddings)
    r, c, d = train_text_embeddings.shape
    assert args.dimension < d, f'dimensionality {args.dimension} must be smaller than {d}'
    train_text_embeddings = train_text_embeddings.reshape((r*c, d))

    # test texts
    test_text_embeddings = [d[:5].cpu().detach().numpy() for d in test_data['texts_embeddings']]
    test_text_embeddings = np.array(test_text_embeddings)
    rt, ct, dt = test_text_embeddings.shape
    print('test shape', rt, ct, dt)
    test_text_embeddings = test_text_embeddings.reshape((rt * ct, dt))

    # text dimensionality reduction
    train_text_embeddings, test_text_embeddings = dimension_reduction(train_text_embeddings, test_text_embeddings, args.dimension)
    train_data['texts_embeddings'] = torch.from_numpy(train_text_embeddings.reshape((r, c, args.dimension)))
    test_data['texts_embeddings'] = torch.from_numpy(test_text_embeddings.reshape((rt, ct, args.dimension)))

    # train images
    train_image_embeddings = [d.squeeze(0).cpu().detach().numpy() for d in train_data['image_embeddings']]
    train_image_embeddings = np.array(train_image_embeddings)
    # test images
    test_image_embeddings = [d.squeeze(0).cpu().detach().numpy() for d in test_data['image_embeddings']]
    test_image_embeddings = np.array(test_image_embeddings)

    # image dimensionality reduction
    train_image_embeddings, test_image_embeddings = dimension_reduction(train_image_embeddings, test_image_embeddings, args.dimension)
    train_data['image_embeddings'] = torch.from_numpy(train_image_embeddings)
    test_data['image_embeddings'] = torch.from_numpy(test_image_embeddings)

    with open(args.output, 'wb') as f:
        print('saving data to {}'.format(args.output))
        pickle.dump(train_data, f)

    with open(args.output.replace('train', 'val'), 'wb') as f:
        print('saving data to {}'.format(args.output.replace('train', 'val')))
        pickle.dump(test_data, f)
