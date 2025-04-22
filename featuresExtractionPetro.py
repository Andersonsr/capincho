import argparse
import pickle
from tqdm import tqdm
import pandas as pd
from foundation_models import model_dict
import torch
import logging

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

    df = pd.read_excel(args.path)
    text_embeddings = []
    image_embeddings = []

    data = {'captions': df['text'].tolist(),
            'image_id': df['cd_guid'].tolist(),
            'image_embeddings': [],
            'text_embeddings': []}

    if args.patched:
        data['patch_embeddings'] = []

    # extraction loop
    for i, row in tqdm(df.iterrows()):
        id = row['cd_guid']
        img_path = f'{args.root}/{id}.png'
        vis_embed = model.visual_embedding(img_path, args.resize)
        txt_embed = model.language_embedding(row['text'])
        data['text_embeddings'].append(txt_embed.detach().cpu())
        data['image_embeddings'].append(vis_embed.detach().cpu())

        if args.patched:
            patches_embeds = model.patch_embedding(img_path)
            logger.debug('Patches embedding shape: {}x{}x{}'.format(
                patches_embeds.shape[0], patches_embeds.shape[1], patches_embeds.shape[2]))

            data['patch_embeddings'].append(patches_embeds)

        logging.debug(f'text embeddings size: {len(text_embeddings)}')
        logging.debug(f'image embeddings size: {len(image_embeddings)}')

    logging.debug(f'caption sample {data["captions"][0]}')

    with open(args.output, 'wb') as f:
        pickle.dump(data, f)



