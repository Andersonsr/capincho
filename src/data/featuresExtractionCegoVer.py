import argparse
import json
import logging
import os
import sys
import pickle
import torch
from tqdm import tqdm

# path trick
path = os.path.normpath(os.path.join(os.path.join(os.path.abspath(__file__)), '..', '..'))
sys.path.append(path)
from models.foundation_models import model_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, required=True, help='input json file')
    parser.add_argument('--output', type=str, required=True,
                        help='output file name, split and extension will be appended')
    parser.add_argument('--model', required=True, type=str, choices=model_dict.keys(), help='encoder model')
    parser.add_argument('--patched', action='store_true', default=False, help='whether to patch image')
    parser.add_argument('--debug', action='store_true', default=False, help='set logging level to debug')
    parser.add_argument('--dataset_root', type=str, default='/mnt/d/PraCegoVer/images', help='dataset images folder')
    args = parser.parse_args()

    assert os.path.exists(args.filename), 'input json file does not exist'
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = logging.getLogger('captioning')
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)

    model = model_dict[args.model](device)
    model.load_model()


    with open(args.filename, 'rb') as f:
        json_object = json.load(f)
        for key in json_object.keys():
            data = {'image_name': [], 'image_id': [], 'image_embeddings': [], 'text_embeddings': [], 'captions': []}
            if args.patched:
                data['patch_embeddings'] = []
                logger.info('patching enabled')

            logging.info(f'processing {key} split')
            for i, image in tqdm(enumerate(json_object[key]), total=len(json_object[key])):
                name = image['filename']
                image_path = os.path.join(args.dataset_root, name)
                caption = image['caption']
                image_embeddings = model.visual_embedding(image_path).detach().cpu()
                text_embeddings = model.language_embedding(caption).detach().cpu()
                print(caption)

                data['image_name'].append(name)
                data['image_id'].append(i)
                data['image_embeddings'].append(image_embeddings)
                data['text_embeddings'].append(text_embeddings)
                data['captions'].append(caption)
                if args.patched:
                    patch_embeddings = model.patch_embedding(image_path).detach().cpu()
                    data['patch_embeddings'].append(patch_embeddings)
                    logging.debug('id {} patch embeddings shape: {}'.format(i, patch_embeddings.shape))

                logging.debug('id {} image embeddings shape: {}'.format(i, image_embeddings.shape))
                logging.debug('text embeddings shape: {}'.format(text_embeddings.shape))

            logging.info('{} split size: {}'.format(key, len(data['image_name'])))
            # save split data
            basename = os.path.basename(args.output).split('.')[0]
            parent_dir = os.path.dirname(args.output)
            key = 'val' if key == 'validation' else key
            with open(f'{parent_dir}/{basename}_{key}.pkl', 'wb') as output:
                pickle.dump(data, output)

