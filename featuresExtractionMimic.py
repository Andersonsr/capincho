import argparse
import json
import logging
import os
import pickle
from tqdm import tqdm
import torch
from foundation_models import model_dict
from torch.utils.data import Dataset
import numpy as np


class MIMICJsonLoader(Dataset):
    def __init__(self, pkl_file, root, model):
        # print(annotation.keys())
        # dict_keys(['id', 'image', 'generate_method', 'reason', 'impression', 'indication', 'history', 'view',
        # 'orientation', 'chexpert_labels', 'conversations'])
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
                   'label': self.labels[index],
                   'findings': self.findings[index],
                   'image': self.image_tensor[index],
                   'image_name': self.image_name[index],
                   'id': self.id[index]}
        return payload

    def get_loader(self, batch_size):
        indices = np.arange(len(self.id))
        sampler = torch.utils.data.SequentialSampler(indices)
        return torch.utils.data.DataLoader(self, batch_size=batch_size, sampler=sampler, shuffle=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, default='/mnt/d/mimic/mimic-cxr-jpg/2.1.0/files',
                        help='Root directory containing the mimic dataset')
    parser.add_argument('--output', type=str, required=True, help='output file name, split and extension will be appended')
    parser.add_argument('--filename', type=str, required=True, help='pkl file containing preprocessed data')
    parser.add_argument('--model', type=str, required=True, help='encoder model used to extract features',
                        choices=model_dict.keys())
    parser.add_argument('--debug', action='store_true', default=False, help='Debug mode')
    parser.add_argument('--resize', action='store_true', default=False, help='Resize image to fit model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    args = parser.parse_args()

    assert os.path.exists(args.filename), 'File does not exist: {}'.format(args.filename)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger = logging.getLogger('captioning')
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    model = model_dict[args.model](device)
    model.load_model()

    data = {'image_name': [], 'image_id': [], 'image_embeddings': [], 'text_embeddings': [], 'captions': [],
            'labels': []}

    json_data = MIMICJsonLoader(args.filename, args.dataset_root, model)
    loader = json_data.get_loader(args.batch_size)

    for batch in tqdm(loader):
        with torch.no_grad():
            image_embeddings = model.visual_embedding(batch['image']).detach().cpu()
            text_embeddings = model.language_embedding(batch['findings']).detach().cpu()
            # print(image_embeddings.shape)
            data['image_embeddings'] += image_embeddings
            data['text_embeddings'] += text_embeddings

    data['image_name'] = json_data.image_name
    data['image_id'] = json_data.id
    data['captions'] = json_data.findings
    data['labels'] = json_data.labels
    data['image_embeddings'] = torch.stack(data['image_embeddings'], dim=0).unsqueeze(dim=1)
    data['text_embeddings'] = torch.stack(data['text_embeddings'], dim=0).unsqueeze(dim=1)
    print(data['image_embeddings'].shape)
    print(data['text_embeddings'].shape)
    with open(args.output, 'wb') as f:
        pickle.dump(data, f)



