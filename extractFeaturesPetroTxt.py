from tqdm import tqdm
import pickle
import pandas as pd
import torch
import logging
from torch.utils.data import Dataset
import argparse
from foundation_models import OpenCLIP, CLIP
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract features from xlsx for image captioning')
    parser.add_argument('--path', '-p', type=str, required=True, help='path to texts xlsx')
    parser.add_argument('--output', '-o', type=str, required=True, help='output path')
    parser.add_argument('--model', '-m', type=str, required=True, help='model name', choices=['openclip', 'clip'])
    parser.add_argument('--debug', '-d', type=bool, default=False, help='debug ')
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = pd.read_excel(args.path)
    logger = logging.getLogger('captioning')
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    if args.model == 'openclip':
        model = OpenCLIP(device)
        model.load_model()

    else:
        model = CLIP(device)
        model.load_model()

    df = pd.read_excel(args.path)
    text_embeddings = []
    for i, row in tqdm(df.iterrows()):
        texts = row['texts']
        txt_embed = model.language_embedding(row['texts'])
        text_embeddings.append(txt_embed.detach().cpu())
        logging.debug(f'text embeddings size: {len(text_embeddings)}')

    data = {'captions': df['texts'].tolist(),
            'text_embeddings': text_embeddings}

    with open(args.output, 'wb') as f:
        pickle.dump(data, f)
