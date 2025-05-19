import argparse
import os
import pickle
from tqdm import tqdm
import pandas as pd
from foundation_models import model_dict
import torch
import logging
from torch.utils.data import Dataset
import numpy as np


class PetroXLSLoader(Dataset):
    def __init__(self,filepath):
        assert os.path.exists(filepath)
        df = pd.read_excel(args.path)
        self.text = df['text'].tolist()
        self.cd_guid = df['cd_guid'].tolist()

    def __len__(self):
        return len(self.cd_guid)

    def __getitem__(self, index):
        return {'text': self.text[index], 'cd_guid': self.cd_guid[index]}

    def get_loader(self, batch_size):
        indices = np.arange(len(self.cd_guid))
        sampler = torch.utils.data.SequentialSampler(indices)
        return torch.utils.data.DataLoader(self, batch_size=batch_size, sampler=sampler, shuffle=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='download and extract features from petro dataset')
    parser.add_argument('--root', type=str, required=True, help='folder containing petro dataset images')
    parser.add_argument('--output', type=str, required=True, help='path to save the extracted features')
    parser.add_argument('--path', type=str, required=True, help='path to xlsx file with text and ids')
    parser.add_argument('--model', type=str, required=True, help='model to use for feature extraction',
                        choices=['clip', 'openclip', 'coca', 'sig-lip'])
    parser.add_argument('--debug', action='store_true', help='debug mode', default=False)
    parser.add_argument('--patched', action='store_true', help='patche images', default=False)
    parser.add_argument('--resize', action='store_true', help='resize images to fit the encoder', default=False)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = logging.getLogger('captioning')
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    # model init
    model = model_dict[args.model](device)
    model.load_model()
    model.backbone.eval()

    dataset_petro = PetroXLSLoader(args.path)
    text_embeddings = []
    image_embeddings = []

    data = {'captions': dataset_petro.text,
            'image_id': dataset_petro.cd_guid,
            'image_embeddings': [],
            'text_embeddings': []}

    if args.patched:
        data['patch_embeddings'] = []

    loader = dataset_petro.get_loader()
    # extraction loop
    for batch in tqdm(loader):
        id = batch['cd_guid']
        img_path = f'{args.root}/{id}.png'
        vis_embed = model.visual_embedding(img_path, args.resize)
        txt_embed = model.language_embedding(batch['text'])
        data['text_embeddings'] += txt_embed.detach().cpu()
        data['image_embeddings'] += vis_embed.detach().cpu()

        logging.debug(f'image embeddings shape: {vis_embed.shape}')
        logging.debug(f'text embeddings shape: {txt_embed.shape}')

    logging.debug(f'caption sample {data["captions"][0]}')
    data['image_embeddings'] = torch.cat(data['image_embeddings']).unsqueeze(dim=1)
    data['text_embeddings'] = torch.cat(data['text_embeddings']).unsqueeze(dim=1)
    with open(args.output, 'wb') as f:
        pickle.dump(data, f)



