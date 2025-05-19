import argparse
import logging
import torch
import os
from pycocotools.coco import COCO
from tqdm import tqdm
import pickle
from foundation_models import model_dict
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# TODO: adicionar pre-processamento para deixar as imagens quadradas antes do resize para funcionar com outros datasets,
#  no dataset petro ja estao quadradas

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['train', 'val'], default='train', help='split to use')
    parser.add_argument('--model', choices=['openclip', 'clip', 'coca', 'sig-lip'], default='openclip', help='model to use')
    parser.add_argument('--save_path', type=str, default='embeddings/coco_train.pkl')
    parser.add_argument('--patched', action='store_true', default=False, help='whether or not to use patches')
    parser.add_argument('--debug', action='store_true', default=False, help='debugging log')
    args = parser.parse_args()
    logger = logging.getLogger('captioning')
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    model = model_dict[args.model](device)
    model.load_model()
    model.backbone.eval()
    coco = COCO(f'datasets_torchvision/coco_2017/annotations/captions_{args.split}2017.json')
    ids = coco.getImgIds()
    imgs = coco.loadImgs(ids)

    data = {'image_name': [], 'image_id': [], 'image_embeddings': [], 'text_embeddings': [], 'captions': []}
    if args.patched:
        data['patch_embeddings'] = []
        logger.info('patching enabled')

    for i, image in enumerate(tqdm(imgs)):
        data['image_name'].append(image['file_name'])
        data['image_id'].append(ids[i])
        img_name = 'datasets_torchvision/coco_2017/{}2017/{}'.format(args.split, image['file_name'])
        img_embeds = model.visual_embedding(img_name)

        logger.debug('image name: {}'.format(image['file_name']))
        logger.debug('img_embeds shape {}x{} '.format(img_embeds.shape[0], img_embeds.shape[1]))

        data['image_embeddings'].append(img_embeds.detach().cpu())
        ann = coco.loadAnns(coco.getAnnIds(ids[i]))
        texts = [e['caption'] for e in ann]

        text_embeds = model.language_embedding(texts[:5])
        data['text_embeddings'].append(text_embeds.detach().cpu())
        data['captions'].append(texts[:5])

        if args.patched:
            patches_embeds = model.patch_embedding(img_name)
            logger.debug('Patches embedding shape: {}x{}x{}'.format(
                patches_embeds.shape[0], patches_embeds.shape[1], patches_embeds.shape[2]))

            data['patch_embeddings'].append(patches_embeds.detach().cpu())

    print(data.keys())
    with open(args.save_path, 'wb') as f:
        pickle.dump(data, f)


