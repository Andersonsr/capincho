import argparse
import logging
import os
import pickle
import json
from tqdm import tqdm
import torch
from foundation_models import model_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, default='/mnt/d/mimic/mimic-cxr-jpg/2.1.0/files',
                        help='Root directory containing the mimic dataset')
    parser.add_argument('--output', type=str, required=True, help='output file name, split and extension will be appended')
    parser.add_argument('--filename', type=str, required=True, help='csv file containing the splits')
    parser.add_argument('--model', type=str, required=True, help='encoder model used to extract features',
                        choices=model_dict.keys())
    parser.add_argument('--debug', action='store_true', default=False, help='Debug mode')
    parser.add_argument('--patched', action='store_true', default=False, help='Compute patched embeddings')
    parser.add_argument('--resize', action='store_true', default=False, help='Resize image to fit model')
    args = parser.parse_args()

    assert os.path.exists(args.filename), 'File does not exist: {}'.format(args.filename)
    logger = logging.getLogger('captioning')
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_dict[args.model](device)
    model.load_model()

    file = open(args.filename, 'r')
    json_data = json.load(file)

    data = {'image_name': [], 'image_id': [], 'image_embeddings': [], 'text_embeddings': [], 'captions': [],
            'labels': []}
    if args.patched:
        data['patch_embeddings'] = []

    for annotation in tqdm(json_data, total=len(json_data)):
        # print(annotation.keys())
        # dict_keys(['id', 'image', 'generate_method', 'reason', 'impression', 'indication', 'history', 'view',
        # 'orientation', 'chexpert_labels', 'conversations'])

        # findings
        texts = [annotation['conversations'][1]['value'].replace('\n', '')]
        impression = annotation['impression']
        if impression is not None:
            texts.append(impression)

        image_name = '/'.join(annotation['image'].split('/')[1:])
        image_path = os.path.join(args.dataset_root, image_name)
        labels = annotation['chexpert_labels']

        if os.path.exists(image_path):
            text_embeddings = model.language_embedding(texts).detach().cpu()
            image_embeddings = model.visual_embedding(image_path).detach().cpu()

            data['image_id'].append(annotation['id'])
            data['image_name'].append(image_name)
            data['image_embeddings'].append(image_embeddings)
            data['text_embeddings'].append(text_embeddings)
            data['labels'].append(labels)
            data['captions'].append(texts)
            if args.patched:
                data['patch_embeddings'].append(model.patch_embedding(image_path).detach().cpu())

        with open(args.output, 'wb') as f:
            pickle.dump(data, f)

