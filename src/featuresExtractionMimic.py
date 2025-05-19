import argparse
import glob
import json
import logging
import os
import pickle
from tqdm import tqdm
import torch
from foundation_models import model_dict
from torch.utils.data import Dataset
import numpy as np


class MIMICChunkLoader(Dataset):
    def __init__(self, pkl_file,):
        assert os.path.exists(pkl_file), '{} does not exist'.format(pkl_file)

        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
            self.id = data['id']
            self.image_name = data['image_name']
            self.image_path = data['image_path']
            self.labels = data['labels']
            self.findings = data['findings']
            self.image_tensor = data['image_tensor']

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        payload = {'image_path': self.image_path[index],
                   'labels': self.labels[index],
                   'findings': self.findings[index],
                   'image': self.image_tensor[index],
                   'image_name': self.image_name[index],
                   'id': self.id[index]}
        return payload

    # torch dataloader cant handle PIL image by default
    def collate_fn(self, batch):
        data = {'image_path': [], 'labels': [], 'findings': [], 'image_name': [], 'id': [], 'image': []}
        for d in batch:
            data['image_path'].append(d['image_path'])
            data['labels'].append(d['labels'])
            data['findings'].append(d['findings'])
            data['image_name'].append(d['image_name'])
            data['id'].append(d['id'])
            data['image'].append(d['image'])

        return data

    def get_loader(self, batch_size):
        indices = np.arange(len(self.id))
        sampler = torch.utils.data.SequentialSampler(indices)
        return torch.utils.data.DataLoader(self, batch_size=batch_size, sampler=sampler, shuffle=False,
                                           collate_fn=self.collate_fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirname', type=str, required=True,
                        help='dir containing chunks')
    parser.add_argument('--output', type=str, required=True, help='dir to save embeddings chunks')
    parser.add_argument('--model', type=str, required=True, help='encoder model used to extract features',
                        choices=model_dict.keys())
    parser.add_argument('--debug', action='store_true', default=False, help='Debug mode')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    chunks = glob.glob(os.path.join(args.dirname, '*.pkl'))

    logger = logging.getLogger('captioning')
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    model = model_dict[args.model](device)
    model.load_model()

    os.makedirs(args.output, exist_ok=True)

    for chunk in chunks:
        data = {'image_name': [], 'image_id': [], 'image_embeddings': [], 'text_embeddings': [], 'captions': [],
                'labels': []}

        logging.info('Loading chunk: {}'.format(chunk))
        json_data = MIMICChunkLoader(chunk)
        loader = json_data.get_loader(args.batch_size)
        logging.debug('Loaded chunk len: {}'.format(len(json_data)))

        for batch in tqdm(loader):
            logging.debug('batch size: {}'.format(len(batch['id'])))
            with torch.no_grad():
                image_embeddings = model.visual_embedding(batch['image']).detach().cpu()
                text_embeddings = model.language_embedding(batch['findings']).detach().cpu()
                data['image_embeddings'] += image_embeddings
                data['text_embeddings'] += text_embeddings
                logging.debug('batch image embeddings shape: {}'.format(image_embeddings.shape))
                logging.debug('batch Text embeddings shape: {}'.format(text_embeddings.shape))

        data['image_name'] = json_data.image_name
        data['image_id'] = json_data.id
        data['captions'] = json_data.findings
        data['labels'] = json_data.labels
        data['image_embeddings'] = torch.stack(data['image_embeddings'], dim=0).unsqueeze(dim=1)
        data['text_embeddings'] = torch.stack(data['text_embeddings'], dim=0).unsqueeze(dim=1)

        logging.debug('image embeddings shape: {}'.format(data['image_embeddings'].shape))
        logging.debug('text embeddings shape: {}'.format(data['text_embeddings'].shape))

        with open(os.path.join(args.output, os.path.basename(chunk)), 'wb') as f:
            pickle.dump(data, f)



