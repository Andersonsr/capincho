import argparse
import json
import logging
import os
import pickle

from tqdm import tqdm
import torch
from foundation_models import model_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', required=True, type=str, help='json to preprocess')
    parser.add_argument('--dataset_root', type=str, default='/mnt/d/mimic/mimic-cxr-jpg/2.1.0/files',
                        help='Root directory containing the mimic dataset')
    parser.add_argument('--output', type=str, required=True, help='output pkl file')
    parser.add_argument('--debug', action='store_true', help='debug mode', default=False)
    parser.add_argument('--model', type=str, required=True, help='encoder model used to extract features',
                        choices=model_dict.keys())
    args = parser.parse_args()

    assert os.path.exists(args.filename), 'File does not exist: {}'.format(args.filename)
    json_file = json.load(open(args.filename, 'r'))
    logger = logging.getLogger('captioning')
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_dict[args.model](device)
    model.load_model()

    logging.info('loading data from {}'.format(args.filename))
    data = {'id': [],
            'image_name': [],
            'image_path': [],
            'labels': [],
            'findings': [],
            'image_tensor': []}

    for i, annotation in tqdm(enumerate(json_file), total=len(json_file)):
        image_name = '/'.join(annotation['image'].split('/')[1:])
        image_path = os.path.join(args.dataset_root, image_name)
        if os.path.exists(image_path):
            image = model.prepare_image(image_path)
            data['id'].append(i)
            data['image_name'].append(image_name)
            data['image_path'].append(image_path)
            data['labels'].append(annotation['chexpert_labels'])
            data['findings'].append(annotation['conversations'][1]['value'].replace('\n', ''))
            data['image_tensor'].append(image.squeeze(0))

    with open(args.output, 'wb') as f:
        pickle.dump(data, f)
