import argparse
import pickle
from tqdm import tqdm
import pandas as pd
import foundation_models
import torch
import logging

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='download and extract features from petro dataset')
    parser.add_argument('--root', type=str, required=True, help='folder containing petro dataset images')
    parser.add_argument('--output', type=str, required=True, help='path to save the extracted features')
    parser.add_argument('--path', type=str, required=True, help='path to xlsx file with text and ids')
    parser.add_argument('--model', type=str, required=True, help='model to use for feature extraction',
                        choices=['clip', 'openclip', 'coca'])
    parser.add_argument('--debug', action='store_true', help='debug mode', default=False)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = logging.getLogger('captioning')
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    # models dict
    model_dict = {'coca': foundation_models.OpenCoCa,
                  'clip': foundation_models.CLIP,
                  'openclip': foundation_models.OpenCLIP}
    # model init
    model = model_dict[args.model](device)
    model.load_model()
    model.backbone.eval()

    df = pd.read_excel(args.path)
    text_embeddings = []
    image_embeddings = []

    # extraction loop
    for i, row in tqdm(df.iterrows()):
        id = row['cd_guid']
        vis_embed = model.visual_embedding(f'{args.root}/{id}.png')
        txt_embed = model.language_embedding(row['text'])
        text_embeddings.append(txt_embed.detach().cpu())
        image_embeddings.append(vis_embed.detach().cpu())

        logging.debug(f'text embeddings size: {len(text_embeddings)}')
        logging.debug(f'image embeddings size: {len(image_embeddings)}')

    data = {'captions': df['text'].tolist(),
            'image_id': df['cd_guid'].tolist(),
            'image_embeddings': image_embeddings,
            'text_embeddings': text_embeddings}

    logging.debug(f'caption sample {data["captions"][0]}')

    with open(args.output, 'wb') as f:
        pickle.dump(data, f)



