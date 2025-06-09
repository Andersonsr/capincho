import argparse
import math
import json
import logging
import os
import sys
import pickle
from tqdm import tqdm
import torch

# path trick
path = os.path.normpath(os.path.join(os.path.join(os.path.abspath(__file__)), '..', '..'))
sys.path.append(path)
from models.foundation_models import prepare_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', required=True, type=str, help='json to preprocess')
    parser.add_argument('--dataset_root', type=str, default='D:/mimic/mimic-cxr-jpg/2.1.0/files',
                        help='Root directory containing the mimic dataset')
    parser.add_argument('--output', type=str, required=True, help='output directory')
    parser.add_argument('--debug', action='store_true', help='debug mode', default=False)
    parser.add_argument('--dim', type=int, default=224, help='output image dimension')
    parser.add_argument('--chunk_size', type=int, default=50000, help='data chunk size')

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    json_file = json.load(open(args.filename, 'r'))
    logger = logging.getLogger('captioning')
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info('loading data from {}'.format(args.filename))
    data = {'id': [],
            'image_name': [],
            'image_path': [],
            'labels': [],
            'findings': [],
            'image_tensor': []}
    
    chunk_counter = 0
    for i, annotation in tqdm(enumerate(json_file), total=len(json_file)):
        image_name = '/'.join(annotation['image'].split('/')[1:])
        image_path = os.path.join(args.dataset_root, image_name)
        if os.path.exists(image_path):
            image = prepare_image(image_path, dim=args.dim, resize=True)
            data['id'].append(i)
            data['image_name'].append(image_name)
            data['image_path'].append(image_path)
            data['findings'].append(annotation['conversations'][1]['value'].replace('\n', ''))
            data['image_tensor'].append(image)
            new_labels = {}
            old_labels = annotation['chexpert_labels']
            for key in old_labels.keys():
                if math.isnan(old_labels[key]):
                    new_labels[key] = 3
                else:
                    new_labels[key] = 2 if old_labels[key] < 0 else old_labels[key]
            data['labels'].append(new_labels)
            logging.debug('image {} shape: {}'.format(i, image.size))

        if (i+1) % args.chunk_size == 0 or i == len(json_file) - 1:
            # save the chunk
            logging.info('chunk {}: {} images'.format(chunk_counter, len(data['image_name'])))
            with open(os.path.join(args.output, f'chunk_{chunk_counter}_{args.dim}.pkl'), 'wb') as f:
                pickle.dump(data, f)

            # start a new chunk
            chunk_counter += 1
            data = {'id': [],
                    'image_name': [],
                    'image_path': [],
                    'labels': [],
                    'findings': [],
                    'image_tensor': []}
