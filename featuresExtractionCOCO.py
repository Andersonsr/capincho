import argparse
import torch
import os
from pycocotools.coco import COCO
from tqdm import tqdm
import pickle
import foundation_models
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['train', 'val'], default='train', help='split to use')
    parser.add_argument('--model', choices=['openclip', 'clip', 'coca'], default='openclip', help='model to use')
    parser.add_argument('--save_path', type=str, default='embeddings/coco_train.pkl')
    args = parser.parse_args()

    model_dict = {'coca': foundation_models.OpenCoCa,
                  'clip': foundation_models.CLIP,
                  'openclip': foundation_models.OpenCLIP,
                  'capivara': foundation_models.Capivara}

    model = model_dict[args.model](device)
    model.load_model()
    model.backbone.eval()
    coco = COCO(f'datasets_torchvision/coco_2017/annotations/captions_{args.split}2017.json')
    ids = coco.getImgIds()
    imgs = coco.loadImgs(ids)

    data = {'image_name': [], 'image_id': [], 'image_embeddings': [], 'text_embeddings': [], 'captions': []}
    for i, image in enumerate(tqdm(imgs)):
        data['image_name'].append(image['file_name'])
        data['image_id'].append(ids[i])
        img_embeds = model.visual_embedding('datasets_torchvision/coco_2017/{}2017/{}'.format(args.split,
                                                                                              image['file_name']))
        data['image_embeddings'].append(img_embeds.detach().cpu())

        ann = coco.loadAnns(coco.getAnnIds(ids[i]))
        texts = [e['caption'] for e in ann]

        text_embeds = model.language_embedding(texts[:5])
        data['text_embeddings'].append(text_embeds.detach().cpu())
        data['captions'].append(texts[:5])

    with open(args.save_path, 'wb') as f:
        pickle.dump(data, f)


