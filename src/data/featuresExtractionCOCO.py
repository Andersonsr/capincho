import argparse
import logging
import torch
import os
import sys
from pycocotools.coco import COCO
from tqdm import tqdm
import pickle

# path trick
path = os.path.normpath(os.path.join(os.path.join(os.path.abspath(__file__)), '..', '..'))
sys.path.append(path)
from models.foundation_models import model_dict

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['train', 'val'], default='train', help='split to use')
    parser.add_argument('--model', choices=model_dict.keys(), default='openclip', help='model to use')
    parser.add_argument('--save_path', type=str, default='embeddings/coco_train.pkl')
    parser.add_argument('--patched', action='store_true', default=False, help='whether or not to use patches')
    parser.add_argument('--debug', action='store_true', default=False, help='debugging log')
    parser.add_argument('--dataset_root', type=str,)
    args = parser.parse_args()
    logger = logging.getLogger('captioning')
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    model = model_dict[args.model](device)
    model.load_model()
    model.backbone.eval()
    coco = COCO(f'{args.dataset_root}/annotations/captions_{args.split}2017.json')
    ids = coco.getImgIds()
    imgs = coco.loadImgs(ids)

    data = {'image_name': [], 'image_id': [], 'image_embeddings': [], 'text_embeddings': [], 'captions': []}

    with torch.no_grad():
        for i, image in enumerate(tqdm(imgs)):
            data['image_name'].append(image['file_name'])
            data['image_id'].append(ids[i])
            img_name = 'E:datasets/coco_2017/{}2017/{}'.format(args.split, image['file_name'])
            img_embeds = model.visual_embedding(img_name)

            logger.debug('image name: {}'.format(image['file_name']))
            logger.debug('img_embeds shape {}x{} '.format(img_embeds.shape[0], img_embeds.shape[1]))

            data['image_embeddings'].append(img_embeds.to(dtype=torch.float32).detach().cpu())
            ann = coco.loadAnns(coco.getAnnIds(ids[i]))
            texts = [e['caption'] for e in ann]

            text_embeds = model.language_embedding(texts[:5])
            data['text_embeddings'].append(text_embeds.to(dtype=torch.float32).detach().cpu())
            data['captions'].append(texts[:5])

    print(data.keys())
    with open(args.save_path, 'wb') as f:
        pickle.dump(data, f)


